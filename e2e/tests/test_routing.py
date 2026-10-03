"""RoutingPolicy implementations, in isolation from RemoteDraftProposer."""
from __future__ import annotations

import pytest

from dsdreal.policy.routing import RoundRobin


def test_round_robin_cycles_in_order():
    rr = RoundRobin()
    picks = [rr.select_drafter(["d0", "d1", "d2"]) for _ in range(7)]
    assert picks == ["d0", "d1", "d2", "d0", "d1", "d2", "d0"]


def test_round_robin_single_drafter_is_a_no_op():
    rr = RoundRobin()
    assert [rr.select_drafter(["d0"]) for _ in range(3)] == ["d0"] * 3


def test_round_robin_rejects_empty_registry():
    with pytest.raises(ValueError, match="no drafters"):
        RoundRobin().select_drafter([])


def test_target_policies():
    from dsdreal.policy.routing import JoinShortestQueue, TargetRoundRobin, make_target_policy
    ids = ["t0", "t1"]
    rr = TargetRoundRobin()
    assert [rr.select_target(ids) for _ in range(4)] == ["t0", "t1", "t0", "t1"]
    jsq = JoinShortestQueue()
    assert jsq.select_target(ids, {"t0": 3, "t1": 1}) == "t1"
    # ties alternate instead of always picking the first
    assert {jsq.select_target(ids, {"t0": 2, "t1": 2}) for _ in range(2)} == {"t0", "t1"}
    assert make_target_policy("random", seed=1).select_target(ids) in ids
