"""Remote EAGLE tree drafter.

The head runs `topK_genrate` (SafeAILab/EAGLE). The target keeps the base
model and the tree-attention verify; only the head and its KV live here.
One head is batch-1, so concurrent sequences swap `stable_kv` around each call.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from dsdreal.draft.eagle_tree_codec import pack_tree, unpack_hidden
from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@dataclass
class _Seq:
    kv: object = None


def load_base_lm_head(base_model: str, *, dtype: torch.dtype, device: str):
    """The tree head calls the base `lm_head` (`head(hidden)` inside topK_genrate).

    Qwen2 does not tie that matrix to the embeddings, so it is loaded on its own.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(base_model, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    shard = os.path.join(base_model, weight_map["lm_head.weight"])
    with safe_open(shard, framework="pt", device="cpu") as fh:
        weight = fh.get_tensor("lm_head.weight")
    vocab, hidden = weight.shape
    layer = torch.nn.Linear(hidden, vocab, bias=False)
    layer.weight.data.copy_(weight)
    return layer.to(dtype=dtype, device=device).eval()


def load_eagle_tree_head(ea_model: str, base_model: str, *, device: str = "cuda",
                         dtype: torch.dtype = torch.bfloat16, total_tokens: int = 60,
                         depth: int = 5, top_k: int = 10, threshold: float = 1.0):
    """Load the EAGLE head only. Embeddings come from `base_model`; the base
    transformer stays on the target."""
    root = os.environ.get("EAGLE_ROOT")
    if not root:
        raise RuntimeError("Set EAGLE_ROOT to the SafeAILab/EAGLE checkout")
    import sys
    if root not in sys.path:
        sys.path.insert(0, root)
    import json

    from eagle.model.cnets1 import Model as EagleHead
    from eagle.model.configs import EConfig

    cfg_path = os.path.join(ea_model, "config.json")
    config = EConfig.from_pretrained(cfg_path)
    with open(cfg_path) as fh:
        bias = json.load(fh).get("bias", True)
    head = EagleHead(config, load_emb=True, path=base_model, bias=bias,
                     total_tokens=total_tokens, depth=depth, top_k=top_k, threshold=threshold)
    weight = os.path.join(ea_model, "pytorch_model.bin")
    state = torch.load(weight, map_location="cpu")
    head.load_state_dict(state, strict=False)
    head.to(dtype=dtype, device=device).eval()
    head.init_tree()
    lm_head = load_base_lm_head(base_model, dtype=dtype, device=device)
    return head, lm_head


class EagleTreeDraftEngine:
    def __init__(self, head, node_id: str = "d0", lm_head=None) -> None:
        self.node_id = node_id
        self.head = head
        self.lm_head = lm_head
        self.state: dict[str, _Seq] = {}
        self.calls = 0

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        (res,) = self.speculate_batch([request])
        if isinstance(res, Exception):
            raise res
        return res

    @torch.no_grad()
    def speculate_batch(self, requests: list[SpeculateRequest]) -> list[SpeculateReply | Exception]:
        t0 = time.perf_counter()
        self.calls += len(requests)
        out: list[SpeculateReply | Exception | None] = [None] * len(requests)
        for i, req in enumerate(requests):
            try:
                out[i] = self._one(req, t0)
            except Exception as exc:  # noqa: BLE001 - one sequence must not kill the batch
                out[i] = exc
        return out  # type: ignore[return-value]

    def _one(self, req: SpeculateRequest, t0: float) -> SpeculateReply:
        seq = self.state.setdefault(req.seq_id, _Seq())
        self.head.stable_kv = seq.kv
        hidden = unpack_hidden(req.hidden_states, req.hidden_shape, req.hidden_dim,
                               req.hidden_dtype or "bfloat16")
        ids = req.token_ids or list(req.prompt_token_ids or [])
        if hidden.numel() == 0 or not ids:
            raise ValueError(f"eagle tree request {req.seq_id} needs hidden states and token ids")
        device = self.head.embed_tokens.weight.device
        hidden = hidden.to(device=device, dtype=self.head.embed_tokens.weight.dtype)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
        draft, retrieve, mask, positions = self.head.topK_genrate(
            hidden, input_ids, self.lm_head, None)
        seq.kv = self.head.stable_kv
        fields = pack_tree(draft, retrieve, mask, positions)
        fields["draft_ms"] = (time.perf_counter() - t0) * 1000.0
        return SpeculateReply(seq_id=req.seq_id, **fields)

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self.state.pop(sid, None)

    def close(self) -> None:
        self.state.clear()
        if hasattr(self.head, "stable_kv"):
            self.head.stable_kv = None
