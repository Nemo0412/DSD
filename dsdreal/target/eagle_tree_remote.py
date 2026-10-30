"""Target side of a remote EAGLE tree.

`RemoteTopK` replaces `ea_layer.topK_genrate`. The base model and
`tree_decoding` stay local; the head runs on a drafter process.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from dsdreal.draft.eagle_tree_codec import pack_hidden, unpack_tree
from dsdreal.proto.messages import SpeculateRequest
from dsdreal.proto.transport import DraftClient


class RemoteTopK:
    """Drop-in for `ea_layer.topK_genrate(hidden, input_ids, head, logits_processor)`."""

    def __init__(self, client: DraftClient, drafter_id: str, seq_id: str) -> None:
        self.client = client
        self.drafter_id = drafter_id
        self.seq_id = seq_id
        self.last_reply = None

    def __call__(self, hidden_states, input_ids, head, logits_processor):
        raw, shape, dim, dtype = pack_hidden(hidden_states)
        req = SpeculateRequest(
            seq_id=self.seq_id,
            drafter_id=self.drafter_id,
            gamma=0,
            token_ids=[int(x) for x in input_ids.view(-1).tolist()],
            hidden_states=raw,
            hidden_dim=dim,
            hidden_dtype=dtype,
            hidden_shape=shape,
        )
        (reply,) = self.client.speculate([req])
        self.last_reply = reply
        return unpack_tree(reply, hidden_states.device)


def load_eagle_base(base_model: str, ea_model: str, *, dtype: str = "bfloat16",
                    total_tokens: int = 60, depth: int = 5, top_k: int = 10,
                    device: int = 0):
    """Base model on this GPU. The local head is moved to CPU; calls go remote."""
    root = os.environ.get("EAGLE_ROOT")
    if not root:
        raise RuntimeError("Set EAGLE_ROOT to the SafeAILab/EAGLE checkout")
    if root not in sys.path:
        sys.path.insert(0, root)
    from eagle.model.ea_model import EaModel

    dt = getattr(torch, dtype)
    model = EaModel.from_pretrained(
        use_eagle3=False,
        base_model_path=base_model,
        ea_model_path=ea_model,
        total_token=total_tokens,
        depth=depth,
        top_k=top_k,
        threshold=1.0,
        torch_dtype=dt,
        low_cpu_mem_usage=True,
        device_map={"": f"cuda:{device}"},
        attn_implementation="eager",
    )
    model.eval()
    model.ea_layer.to("cpu")
    torch.cuda.empty_cache()
    return model


def generate(model, remote: RemoteTopK, input_ids: torch.Tensor, max_new_tokens: int,
             max_length: int = 2048) -> tuple[list[int], list[dict]]:
    """One sequence. Returns new token ids and one record per verified round.

    Each record is the round that produced those tokens: num_confirmed is the
    length before the verify, n_draft is the tree width, accept_len is the
    accepted path (not counting the bonus token).
    """
    root = os.environ["EAGLE_ROOT"]
    if root not in sys.path:
        sys.path.insert(0, root)
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import (
        evaluate_posterior,
        initialize_tree,
        reset_tree_mode,
        tree_decoding,
        update_inference_inputs,
    )

    model.ea_layer.topK_genrate = remote
    if hasattr(model.ea_layer, "reset_kv"):
        model.ea_layer.reset_kv()
    input_ids = input_ids.to(model.base_model.device).view(1, -1).clone()
    prompt_len = int(input_ids.shape[1])
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
        model.base_model, max_length=max_length)
    reset_tree_mode(model)
    model.ea_layer.reset_kv()

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _logits, _hidden, _sample = initialize_tree(
        input_ids, model, past_key_values, None)
    first = remote.last_reply
    rounds: list[dict] = []
    new_token = 0
    hard_max = max_length - model.ea_layer.total_tokens - 10

    for _ in range(hard_max):
        proposed = remote.last_reply
        confirmed = int(input_ids.shape[1])
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(input_ids.device)
        _logits, hidden_state_new, _outputs = tree_decoding(
            model, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices)
        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]
        best_candidate, accept_length, sample_p = evaluate_posterior(_logits, candidates, None)
        accept_len = int(accept_length.item() if hasattr(accept_length, "item") else accept_length)
        n_draft = max(int(draft_tokens.shape[-1]) - 1, accept_len)
        verified_ms = time.perf_counter() * 1000.0
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, _hidden, _sample = update_inference_inputs(
            input_ids, candidates, best_candidate, accept_length, retrieve_indices, None, new_token,
            past_key_values_data, current_length_data, model, hidden_state_new, sample_p)
        rounds.append({
            "num_confirmed": confirmed,
            "n_draft": n_draft,
            "accept_len": accept_len,
            "verified_ms": verified_ms,
            "reply": proposed or first,
        })
        if new_token >= max_new_tokens or input_ids.shape[1] > hard_max:
            break

    produced = [int(t) for t in input_ids[0, prompt_len:].tolist()]
    if len(produced) > max_new_tokens:
        produced = produced[:max_new_tokens]
    return produced, rounds
