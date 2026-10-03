"""Draft engine that replays a pre-recorded token continuation per prompt.

No model, no GPU. `speculate()` doesn't get a real request id yet (Stage 0
has no transport-level identity channel wired up on the target side), so it
identifies a sequence by matching a prefix of `prompt_token_ids` against the
known trace prompts -- longer prompts are tried first since a longer exact
match is strictly more specific. Once matched, `seq_id` is trusted for the
rest of that sequence's life (`free()` drops it).

Position within the recorded continuation comes from `num_confirmed`, which
`SpeculateRequest` documents as the absolute confirmed length -- so replies
are correct even under retry, and this engine never needs to track
speculative tails the way a stateful KV-owning drafter would (see
EchoDraftEngine).
"""
from __future__ import annotations

from dataclasses import dataclass

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@dataclass
class _SeqTrace:
    prompt_len: int
    continuation: list[int]


class ReplayDraftEngine:
    """`traces` maps a prompt's token ids (tuple) to its recorded
    continuation (the tokens a real run produced after that prompt)."""

    def __init__(self, traces: dict[tuple[int, ...], list[int]],
                 node_id: str = "d0") -> None:
        self.node_id = node_id
        self._traces = sorted(traces.items(), key=lambda kv: -len(kv[0]))
        self._state: dict[str, _SeqTrace] = {}
        self.calls = 0
        self.tokens_proposed = 0

    def _match(self, history: list[int]) -> _SeqTrace:
        for prompt, continuation in self._traces:
            n = len(prompt)
            if len(history) >= n and tuple(history[:n]) == prompt:
                return _SeqTrace(prompt_len=n, continuation=continuation)
        # Unknown prompt: nothing to replay for it. Cached as a permanent
        # miss so we don't re-scan every step.
        return _SeqTrace(prompt_len=len(history), continuation=[])

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        self.calls += 1
        seq = self._state.get(request.seq_id)
        if seq is None:
            if not request.prompt_token_ids:
                return SpeculateReply(seq_id=request.seq_id, draft_tokens=[])
            seq = self._match(request.prompt_token_ids)
            self._state[request.seq_id] = seq

        gen_pos = request.num_confirmed - seq.prompt_len
        if gen_pos < 0 or not seq.continuation:
            return SpeculateReply(seq_id=request.seq_id, draft_tokens=[])
        draft = seq.continuation[gen_pos: gen_pos + request.gamma]
        self.tokens_proposed += len(draft)
        return SpeculateReply(seq_id=request.seq_id, draft_tokens=list(draft))

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self._state.pop(sid, None)

    def close(self) -> None:
        pass
