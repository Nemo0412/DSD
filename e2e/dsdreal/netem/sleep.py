"""Delay-element emulator — the DSD-Sim network model, reproduced.

The paper models links as "delay elements associated with send and receive
events, parameterized by configurable RTT and jitter". This is that, which is
why it is the DEFAULT rather than tc netem: matching the simulator's own
abstraction is the point, and a real qdisc would add congestion control and
queueing that the simulator never modelled.
"""
from __future__ import annotations

import random
from collections.abc import Mapping


class SleepEmulator:
    """Half the RTT on send, half on receive, plus jitter.

    `matrix` overrides `rtt_ms` per node pair, so a fat-tree latency matrix
    (same-rack / cross-rack / cross-pod) drops straight in.
    """

    name = "sleep"

    def __init__(
        self,
        rtt_ms: float = 0.0,
        jitter_ms: float = 0.0,
        jitter_dist: str = "normal",
        matrix: Mapping[tuple[str, str], float] | None = None,
        seed: int = 0,
    ) -> None:
        self.rtt_ms = rtt_ms
        self.jitter_ms = jitter_ms
        self.jitter_dist = jitter_dist
        self.matrix = dict(matrix or {})
        self._rng = random.Random(seed)

    def base_rtt_ms(self, src: str, dst: str) -> float:
        if (src, dst) in self.matrix:
            return self.matrix[(src, dst)]
        if (dst, src) in self.matrix:
            return self.matrix[(dst, src)]
        return self.rtt_ms

    def one_way_ms(self, src: str, dst: str) -> float:
        half = self.base_rtt_ms(src, dst) / 2.0
        if not self.jitter_ms or self.jitter_dist == "none":
            return max(half, 0.0)
        if self.jitter_dist == "uniform":
            j = self._rng.uniform(-self.jitter_ms, self.jitter_ms) / 2.0
        else:  # normal, truncated at zero
            j = self._rng.gauss(0.0, self.jitter_ms) / 2.0
        return max(half + j, 0.0)

    def install(self) -> None:
        pass
