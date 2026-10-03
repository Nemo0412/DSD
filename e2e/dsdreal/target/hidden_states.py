"""Export the target's EAGLE3 aux hidden states to a custom_class proposer.

custom_class gets only token ids (gpu_model_runner.py, the custom_class
branch of propose_draft_token_ids), and the runner only asks the model for
aux hidden states when the method is eagle3. A remote EAGLE3 head needs
them, so two patches on GPUModelRunner, both inert unless the drafter sets
`wants_aux_hidden_states = True`:

1. after __init__: turn on `use_aux_hidden_state_outputs`, so load_model
   runs `_setup_eagle3_aux_hidden_state_outputs` and the model returns
   (hidden, [aux_0, aux_1, aux_2]) -- the same default layers vLLM's own
   EAGLE3 uses.
2. around propose_draft_token_ids: hand the drafter, per input-batch row, the
   aux hidden states of the positions that became *confirmed* this step,
   concatenated along the feature dim ([n, 3 * hidden], what EAGLE3's fc
   consumes):
   - a row with scheduled drafts (verify step): the first len(sampled)
     positions = the previous last token + the accepted drafts;
   - any other row (prefill chunk, or a plain 1-token decode): every
     scheduled position.
   So for every request the positions delivered so far always number
   num_tokens_no_spec - 1: every confirmed position except the newest token,
   whose hidden state the next step computes. RemoteDraftProposer asserts
   exactly that (the K8 live check), which is what catches a silent
   misalignment -- the failure mode here is wrong features, not a crash.

Flattened token order is input_batch.req_ids order (gpu_model_runner.py,
`tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]`), and
rows are not reordered between the forward and propose within a step.

Requires enable_prefix_caching=False on the target: a prefix-cache hit skips
the forward over the cached positions, so their hidden states never exist.
"""
from __future__ import annotations

import inspect

_installed = False


def install() -> None:
    """Idempotent. Call before the engine is constructed, in the process that
    runs the model (in-process engine: the driver itself)."""
    global _installed
    if _installed:
        return
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as R

    orig_init = R.__init__
    orig_propose = R.propose_draft_token_ids
    sig = inspect.signature(orig_propose)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if getattr(getattr(self, "drafter", None), "wants_aux_hidden_states", False):
            if self.cache_config.enable_prefix_caching:
                # A cache hit skips the forward over the cached prefix: no aux
                # hidden states for those positions, and the drafter would be
                # short of features (seen on GPU: shared chat-template prefix).
                raise RuntimeError("aux hidden-state export needs enable_prefix_caching=False")
            self.use_aux_hidden_state_outputs = True

    def propose_draft_token_ids(self, *args, **kwargs):
        drafter = self.drafter
        if getattr(drafter, "wants_aux_hidden_states", False):
            b = sig.bind(self, *args, **kwargs).arguments
            aux = b["aux_hidden_states"]
            if aux is None:
                raise RuntimeError("wants_aux_hidden_states but the model returned none")
            drafter.receive_hidden_states(_new_positions(
                self, b["scheduler_output"], b["sampled_token_ids"], aux, torch))
        return orig_propose(self, *args, **kwargs)

    R.__init__ = __init__
    R.propose_draft_token_ids = propose_draft_token_ids
    _installed = True


def _new_positions(runner, scheduler_output, sampled, aux, torch):
    """-> {row: (req_id, cpu tensor [n, 3 * hidden])} for rows with n > 0."""
    req_ids = runner.input_batch.req_ids
    spec = scheduler_output.scheduled_spec_decode_tokens
    rows, idx, off = [], [], 0
    for i, rid in enumerate(req_ids):
        n = scheduler_output.num_scheduled_tokens.get(rid, 0)
        valid = len(sampled[i]) if rid in spec else n
        if valid:
            rows.append((i, rid, len(idx), valid))
            idx.extend(range(off, off + valid))
        off += n
    if not idx:
        return {}
    feats = torch.cat(aux, dim=-1)
    sel = feats.index_select(0, torch.tensor(idx, device=feats.device)).to("cpu")
    return {i: (rid, sel[start:start + valid]) for i, rid, start, valid in rows}
