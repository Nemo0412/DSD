"""Wire messages for the draft <-> target link.

Kept codec-agnostic on purpose: these are msgspec Structs, so they encode to
msgpack for ZMQ, but nothing outside this module depends on that choice.
Payload is deliberately tiny — token ids only, tens of bytes per request —
because DSD-Sim models the link as pure latency, not bandwidth.
"""
from __future__ import annotations

import msgspec


class SpeculateRequest(msgspec.Struct, omit_defaults=True):
    """Ask a drafter for `gamma` tokens on top of the confirmed prefix."""

    seq_id: str
    drafter_id: str
    gamma: int
    # Tokens the target confirmed since the drafter last heard from us. The
    # drafter appends these and drops anything it speculated beyond them.
    accepted_tokens: list[int] = []
    # Total confirmed length after applying `accepted_tokens`. The drafter
    # rolls its own state back to exactly this length; carrying the absolute
    # value makes the protocol idempotent under retry.
    num_confirmed: int = 0
    # Sent once, on the first request for a sequence.
    prompt_token_ids: list[int] | None = None
    # EAGLE3 only: the target's aux hidden states for the positions confirmed
    # since the last request (all prompt positions on the first one), row-major
    # [n, hidden_dim], raw `hidden_dtype` bytes. After this request the
    # drafter holds features for exactly num_confirmed - 1 positions.
    hidden_states: bytes | None = None
    hidden_dim: int = 0
    hidden_dtype: str = ""


class SpeculateReply(msgspec.Struct, omit_defaults=True):
    """Draft tokens, plus enough timing to separate compute from network."""

    seq_id: str
    draft_tokens: list[int] = []
    # Drafter-side compute, measured by the drafter itself.
    draft_ms: float = 0.0
    # Filled in client-side by the transport; never crosses the wire.
    sent_ms: float = 0.0
    recv_ms: float = 0.0
    injected_delay_ms: float = 0.0

    @property
    def rtt_observed_ms(self) -> float:
        return self.recv_ms - self.sent_ms

    @property
    def net_excess_ms(self) -> float:
        """Round trip minus what we asked for and minus drafter compute.

        This is the real serialization + queueing component: the part the
        simulator does not model. Reported so transport choice stays an
        observable rather than an assumption.
        """
        return self.rtt_observed_ms - self.injected_delay_ms - self.draft_ms


class FreeSeq(msgspec.Struct):
    """Release drafter-side state for finished sequences."""

    seq_ids: list[str]
