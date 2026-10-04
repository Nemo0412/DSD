"""Raw per-round and per-request records -- the only things the collector reads.

Both sides stamp times as `time.perf_counter() * 1000` (absolute ms), the same
clock the transport already uses for `SpeculateReply.sent_ms/recv_ms`. That
is only meaningful because the driver runs the engine in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0): proposer and driver share one clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RoundRecord:
    """One `speculate()` call for one sequence, as seen from the target.

    Whether the proposal was accepted is not known yet at this point -- it
    shows up as the growth of `num_confirmed` at the sequence's next round,
    which is how the collector derives it.
    """

    seq_id: str
    drafter_id: str
    gamma: int                 # requested
    n_draft: int               # actually returned (<= gamma)
    num_confirmed: int         # confirmed length when this round was proposed
    t_send_ms: float
    t_recv_ms: float
    draft_ms: float
    injected_ms: float
    # Only on a sequence's first round: its full confirmed history (prompt +
    # the prefill-sampled token), which is how the collector ties seq_id back
    # to the driver's request_id.
    prompt_token_ids: list[int] | None = None


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    arrival_ms: float          # scheduled arrival, absolute
    prompt_token_ids: list[int]
    max_tokens: int
    token_ts_ms: list[float] = field(default_factory=list)  # one per output token
    finish_ms: float | None = None
    output_token_ids: list[int] = field(default_factory=list)
    # Client-side scenario (dsdSim's): the request is born at the drafter's
    # client, reaches the target `uplink_ms` later, and every token reaches
    # the client `downlink_ms` after the target produced it. token_ts_ms and
    # finish_ms are client-side times; round timestamps stay target-side.
    uplink_ms: float = 0.0
    downlink_ms: float = 0.0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.token_ts_ms)
