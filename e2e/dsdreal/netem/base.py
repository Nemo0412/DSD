"""Network delay injection, independent of how bytes actually move.

Transport (ZMQ, in-process, ...) and emulation (how much delay) are separate
axes. Keeping them apart lets us run ZMQ with zero injected delay to isolate
IPC overhead, and run the in-process path with realistic RTT so Stage 2 can
already produce latency numbers.

Everything here is BLOCKING: vLLM calls propose() synchronously inside the
engine step, so there is no event loop to await on.
"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

# time.sleep overshoots by roughly this much on Linux/WSL; spin the tail.
_SPIN_MARGIN_S = 2e-4


def block_for(seconds: float) -> None:
    """Sleep accurately enough for sub-millisecond delays."""
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    coarse = seconds - _SPIN_MARGIN_S
    if coarse > 0:
        time.sleep(coarse)
    while time.perf_counter() < deadline:
        pass


@runtime_checkable
class NetworkEmulator(Protocol):
    """One-way delay between two nodes in the (virtual) topology."""

    def one_way_ms(self, src: str, dst: str) -> float: ...

    def install(self) -> None:
        """Apply out-of-band setup (tc qdiscs). No-op for delay-injection."""
