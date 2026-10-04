"""Deterministic drafter with no model and no GPU.

Exists so transport, fan-out, rollback and timing can be tested without
loading weights. Token ids are a pure function of (seq_id, position), so two
runs agree and rollback is observable: re-drafting from the same confirmed
length must reproduce the same tokens.

It models the state machine faithfully, which is the point. After proposing
gamma tokens a real drafter's KV extends gamma past the confirmed prefix — it
ran forward passes over them. The next round the target reports how many
survived, and the drafter must drop the speculative tail before appending the
target's authoritative tokens. Those are not the same thing as the tokens it
proposed: the last one is the target's bonus or correction token, which the
drafter never saw.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@dataclass
class _SeqState:
    confirmed: list[int] = field(default_factory=list)
    # Tokens speculated last round and not yet confirmed. In a paged-KV engine
    # this is a logical length, not a copy: the blocks stay allocated and the
    # rejected entries are overwritten by the next slot_mapping.
    speculative: int = 0


class EchoDraftEngine:
    def __init__(self, node_id: str = "d0", vocab_size: int = 151936,
                 compute_ms_per_token: float = 0.0) -> None:
        self.node_id = node_id
        self.vocab_size = vocab_size
        self.compute_ms_per_token = compute_ms_per_token
        self.state: dict[str, _SeqState] = {}
        self.rollbacks = 0
        self.rolled_back_tokens = 0

    def _token(self, seq_id: str, pos: int) -> int:
        return (hash((seq_id, pos)) & 0x7FFFFFFF) % self.vocab_size

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        t0 = time.perf_counter()
        seq = self.state.get(request.seq_id)
        if seq is None:
            seq = _SeqState(confirmed=list(request.prompt_token_ids or []))
            self.state[request.seq_id] = seq

        # 1. Drop the speculative tail from the previous round.
        if seq.speculative:
            self.rollbacks += 1
            self.rolled_back_tokens += seq.speculative
            seq.speculative = 0

        # 2. Adopt the target's authoritative tokens.
        seq.confirmed.extend(request.accepted_tokens)

        # 3. num_confirmed is absolute, which makes the protocol idempotent
        #    under retry and catches desync loudly instead of silently.
        if request.num_confirmed and len(seq.confirmed) != request.num_confirmed:
            raise ValueError(
                f"drafter desync on {request.seq_id}: local "
                f"{len(seq.confirmed)} != target {request.num_confirmed}"
            )

        # 4. Speculate.
        base = len(seq.confirmed)
        tokens = [self._token(request.seq_id, base + i) for i in range(request.gamma)]
        seq.speculative = len(tokens)

        if self.compute_ms_per_token:
            from dsdreal.netem.base import block_for
            block_for(self.compute_ms_per_token * request.gamma / 1000.0)
        return SpeculateReply(
            seq_id=request.seq_id,
            draft_tokens=tokens,
            draft_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self.state.pop(sid, None)

    def close(self) -> None:
        self.state.clear()
