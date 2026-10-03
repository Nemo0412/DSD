"""Which drafter a fresh request gets sent to.

Decided once, at the "fresh" row in RemoteDraftProposer.propose() (a new
request occupying a slot), and stuck to for that request's whole lifetime.
A drafter tracks confirmed length per seq_id (VllmDraftEngine, EchoDraftEngine
-- see their `num_confirmed` desync check); switching drafters mid-request
would hand a later round to a drafter that never saw the earlier ones.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RoutingPolicy(Protocol):
    def select_drafter(self, drafter_ids: list[str]) -> str: ...


class RoundRobin:
    """Cycles through `drafter_ids` in the order given. Stateful only in
    that one counter, so N requests arriving together (a single propose()
    batch) still spread round-robin rather than piling onto drafter_ids[0]."""

    def __init__(self) -> None:
        self._next = 0

    def select_drafter(self, drafter_ids: list[str]) -> str:
        if not drafter_ids:
            raise ValueError("no drafters to route to")
        drafter_id = drafter_ids[self._next % len(drafter_ids)]
        self._next += 1
        return drafter_id


# --------------------------------------------------------------------------- targets
# Request -> target, decided once at admission by the router frontend
# (dsdreal/router/frontend.py), the paper's routing policies. `needs_depths`
# tells the router whether to poll every target's live queue depth
# (waiting + running, from the target itself) before each decision.


class TargetRoundRobin:
    needs_depths = False

    def __init__(self) -> None:
        self._next = 0

    def select_target(self, target_ids: list[str], depths: dict[str, int] | None = None) -> str:
        t = target_ids[self._next % len(target_ids)]
        self._next += 1
        return t


class TargetRandom:
    needs_depths = False

    def __init__(self, seed: int = 0) -> None:
        import random
        self._rng = random.Random(seed)

    def select_target(self, target_ids: list[str], depths: dict[str, int] | None = None) -> str:
        return self._rng.choice(target_ids)


class JoinShortestQueue:
    """Fewest outstanding requests; ties go round-robin so equal queues do not
    all pile onto target_ids[0]."""
    needs_depths = True

    def __init__(self) -> None:
        self._tie = 0

    def select_target(self, target_ids: list[str], depths: dict[str, int] | None = None) -> str:
        if not depths:
            raise ValueError("JSQ needs queue depths")
        low = min(depths[t] for t in target_ids)
        ties = [t for t in target_ids if depths[t] == low]
        t = ties[self._tie % len(ties)]
        self._tie += 1
        return t


def make_target_policy(name: str, seed: int = 0):
    return {"rr": TargetRoundRobin, "random": lambda: TargetRandom(seed),
            "jsq": JoinShortestQueue}[name]()
