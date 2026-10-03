"""Draft engine backed by a real vLLM model.

Stage 2a scope (option A from the design discussion): stateless per-round
`generate()` over the full confirmed prefix, relying on `enable_prefix_caching`
(a stable, public engine flag) to make repeated calls over a growing prefix
cheap. No vLLM internals are touched -- deliberately, since anything more
"efficient" (a persistent, pausable, rollback-aware draft-side session) would
mean reimplementing the target's own KV-rollback scheduling against
version-sensitive internals for a real drafter that, on every rejection round,
would have to fall back to resubmission anyway. See the Stage 2a design
discussion for the full argument.

Correctness rests entirely on determinism: fp32 + `enforce_eager` + greedy
(temperature=0) makes every `generate()` call over the same token prefix
reproduce the same continuation regardless of whether it hit the prefix
cache, so state here is just bookkeeping, never a source of divergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@dataclass
class _SeqState:
    confirmed: list[int] = field(default_factory=list)


class VllmDraftEngine:
    """Wraps an already-constructed `vllm.LLM` (the caller owns its
    lifecycle -- sizing/dtype/gpu_memory_utilization are a construction-time
    decision made once, not this class's concern)."""

    def __init__(self, llm, node_id: str = "d0") -> None:
        self.node_id = node_id
        self.llm = llm
        self.state: dict[str, _SeqState] = {}
        self.calls = 0

    def _advance(self, request: SpeculateRequest) -> _SeqState:
        """Adopt the target's confirmed tokens; raise on desync."""
        seq = self.state.get(request.seq_id)
        if seq is None:
            seq = _SeqState(confirmed=list(request.prompt_token_ids or []))
            self.state[request.seq_id] = seq

        seq.confirmed.extend(request.accepted_tokens)
        if request.num_confirmed and len(seq.confirmed) != request.num_confirmed:
            raise ValueError(
                f"drafter desync on {request.seq_id}: local "
                f"{len(seq.confirmed)} != target {request.num_confirmed}"
            )
        return seq

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        (result,) = self.speculate_batch([request])
        if isinstance(result, Exception):
            raise result
        return result

    def speculate_batch(self, requests: list[SpeculateRequest]
                        ) -> list[SpeculateReply | Exception]:
        """One generate() over every sequence in the batch.

        Per-sequence results come back in order; a desync on one sequence is
        that sequence's error and does not fail the rest. Batching can shift
        bf16 draft numerics (batch-dependent reductions), which only changes
        *which* tokens get proposed -- the target's verify keeps the output
        lossless either way.
        """
        import time

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.calls += len(requests)
        results: list[SpeculateReply | Exception | None] = [None] * len(requests)
        prompts, params, rows = [], [], []
        for i, request in enumerate(requests):
            try:
                seq = self._advance(request)
            except Exception as exc:  # noqa: BLE001 - this sequence's error only
                results[i] = exc
                continue
            if request.gamma <= 0:
                results[i] = SpeculateReply(seq_id=request.seq_id, draft_tokens=[])
                continue
            prompts.append(TokensPrompt(prompt_token_ids=seq.confirmed))
            params.append(SamplingParams(temperature=0.0, max_tokens=request.gamma))
            rows.append(i)

        if prompts:
            t0 = time.perf_counter()
            outs = self.llm.generate(prompts, params, use_tqdm=False)
            # One forward per step serves all of them; each reply carries the
            # batch's wall time, which is what that sequence actually waited.
            draft_ms = (time.perf_counter() - t0) * 1000.0
            for i, out in zip(rows, outs):
                results[i] = SpeculateReply(
                    seq_id=requests[i].seq_id,
                    draft_tokens=list(out.outputs[0].token_ids),
                    draft_ms=draft_ms,
                )
        return results  # type: ignore[return-value]

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self.state.pop(sid, None)

    def close(self) -> None:
        self.state.clear()
