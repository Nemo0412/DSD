"""What a drafter must implement, regardless of where it runs."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@runtime_checkable
class DraftEngine(Protocol):
    """Owns draft-side model state and KV for a set of sequences.

    Implementations: EchoDraftEngine (no GPU, for plumbing tests),
    VllmDraftEngine (a real vLLM engine), ReplayDraftEngine (trace replay).
    """

    node_id: str

    def speculate(self, request: SpeculateRequest) -> SpeculateReply: ...

    def free(self, seq_ids: list[str]) -> None: ...

    def close(self) -> None: ...


def speculate_many(engine, requests: list[SpeculateRequest]) -> list[SpeculateReply | Exception]:
    """Serve a batch, per-request results in order; an exception in one slot
    is that request's error, never the batch's.

    Engines that can batch (VllmDraftEngine: one generate() over every
    sequence) expose `speculate_batch` with this same contract. The rest are
    served one by one -- fine for Echo/Replay, but a real model served that
    way pays a full engine call per sequence, which at N sequences per step
    is exactly the serialisation the transport's batch contract exists to
    avoid.
    """
    batch = getattr(engine, "speculate_batch", None)
    if batch is not None:
        return batch(requests)
    out: list[SpeculateReply | Exception] = []
    for r in requests:
        try:
            out.append(engine.speculate(r))
        except Exception as exc:  # noqa: BLE001 - becomes this request's error reply
            out.append(exc)
    return out
