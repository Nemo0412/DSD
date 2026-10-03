"""Speculation-window (gamma) policies, chosen once per request.

Mirrors dsdSim's GammaPolicy (src/sim.py): the simulator calls
`select_gamma` once per conversation, before its first round, and keeps that
gamma for every round -- so does RemoteDraftProposer (on a sequence's fresh
round). vLLM's `num_speculative_tokens` is then only the upper bound.

`SpecPPGamma` is a line-for-line port of dsdSim's `SpecPPGammaPolicy`. Its
input there is the acceptance regressor's per-position *conditional*
acceptance probabilities for the conversation's context length; here it is
an `AcceptanceProfile` measured on the real pair (a vanilla calibration run,
see `AcceptanceProfile.from_steps`), bucketed by context length the same way.
"""
from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class WindowPolicy(Protocol):
    def select_gamma(self, context_len: int) -> int: ...


@dataclass(frozen=True)
class StaticGamma:
    gamma: int

    def select_gamma(self, context_len: int) -> int:
        return self.gamma


@dataclass
class AcceptanceProfile:
    """Per-position conditional acceptance, p[k] = P(draft k+1 accepted |
    drafts 1..k accepted), per context-length bucket.

    `edges` split context lengths into len(edges)+1 buckets; bucket i covers
    [edges[i-1], edges[i]).
    """

    edges: list[int] = field(default_factory=list)
    probs: list[list[float]] = field(default_factory=lambda: [[]])

    def position_probabilities(self, context_len: int) -> list[float]:
        return list(self.probs[bisect.bisect_right(self.edges, context_len)])

    @classmethod
    def from_steps(cls, rows: Iterable[Mapping], context_len: Mapping[str, int],
                   edges: Sequence[int] = (), min_count: int = 20) -> "AcceptanceProfile":
        """Estimate from steps.jsonl rows (dsdreal/metrics/export.py).

        A round with n_draft drafts and `accepted` accepted observes position
        k (1-based) iff k <= accepted + 1 and k <= n_draft -- verification
        stops at the first rejection -- and accepts it iff k <= accepted.
        `context_len` maps request_id -> prompt length (the simulator's
        context feature). Positions seen fewer than `min_count` times in a
        bucket inherit the pooled estimate, then the previous position's.
        """
        edges = sorted(edges)
        nb = len(edges) + 1
        depth = 0
        att: list[dict[int, int]] = [dict() for _ in range(nb + 1)]  # last = pooled
        acc: list[dict[int, int]] = [dict() for _ in range(nb + 1)]
        for r in rows:
            b = bisect.bisect_right(edges, context_len[r["request_id"]])
            n, a = int(r["n_draft"]), int(r["accepted"])
            depth = max(depth, n)
            for k in range(1, min(a + 1, n) + 1):
                for bb in (b, nb):
                    att[bb][k] = att[bb].get(k, 0) + 1
                    if k <= a:
                        acc[bb][k] = acc[bb].get(k, 0) + 1

        def est(bb: int, k: int) -> float | None:
            n = att[bb].get(k, 0)
            return acc[bb].get(k, 0) / n if n >= min_count else None

        pooled: list[float] = []
        for k in range(1, depth + 1):
            p = est(nb, k)
            pooled.append(p if p is not None else (pooled[-1] if pooled else 0.0))
        probs = []
        for b in range(nb):
            row = []
            for k in range(1, depth + 1):
                p = est(b, k)
                row.append(p if p is not None else pooled[k - 1])
            probs.append(row)
        return cls(edges=list(edges), probs=probs)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"edges": self.edges, "probs": self.probs}, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "AcceptanceProfile":
        d = json.loads(Path(path).read_text())
        return cls(edges=list(d["edges"]), probs=[list(p) for p in d["probs"]])


class SpecPPGamma:
    """dsdSim SpecPPGammaPolicy.select_gamma, ported verbatim: grow gamma
    while the probability that some draft so far is rejected,
    1 - prod(p_1..p_gamma), stays <= stop_threshold; clamp to [min, max]."""

    def __init__(self, profile: AcceptanceProfile | None, min_gamma: int = 2,
                 max_gamma: int = 8, stop_threshold: float = 0.7,
                 fallback_gamma: int = 4) -> None:
        if not 0.0 <= stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be within [0, 1]")
        self.profile = profile
        self.min = max(1, int(min_gamma))
        self.max = max(self.min, int(max_gamma))
        self.threshold = float(stop_threshold)
        self.fallback = max(1, int(fallback_gamma))

    def select_gamma(self, context_len: int) -> int:
        if self.profile is None:
            return self.fallback
        probabilities = self.profile.position_probabilities(context_len)
        if not probabilities:
            return self.fallback
        while len(probabilities) < self.max:
            probabilities.append(probabilities[-1])

        min_tokens, max_tokens = self.min, self.max
        cum_accept = 1.0
        selected = min_tokens
        for idx in range(1, max_tokens + 1):
            prob = max(0.0, min(1.0, float(probabilities[idx - 1])))
            cum_accept *= prob
            if idx < min_tokens:
                selected = idx
                continue
            if 1.0 - cum_accept > self.threshold:
                selected = max(min_tokens, idx - 1)
                break
            selected = idx
        return max(min_tokens, min(max_tokens, selected))
