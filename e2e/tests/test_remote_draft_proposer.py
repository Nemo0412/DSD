"""RemoteDraftProposer against a real DraftEngine (EchoDraftEngine), no GPU
and no vLLM import -- `propose()`'s inputs are hand-built numpy arrays
shaped like what gpu_model_runner.py actually passes.

Regression coverage for a real bug: propose() built SpeculateRequest without
ever setting accepted_tokens, so a stateful drafter's local confirmed count
silently fell behind num_confirmed and desynced a few rounds in.
ReplayDraftEngine never noticed (it's purely positional off num_confirmed),
which is exactly why this stayed hidden until VllmDraftEngine hit it on GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.policy.routing import RoundRobin
from dsdreal.proto.transport import InProcessDraftClient
from dsdreal.target import proposer


class _FakeSpecConfig:
    def __init__(self, k):
        self.num_speculative_tokens = k


class _FakeVllmConfig:
    def __init__(self, k):
        self.speculative_config = _FakeSpecConfig(k)


def _batch(rows, max_len=32):
    """rows: list of (sampled_this_step, full_history) -> the three arrays
    propose() receives. `sampled_this_step` empty means "not active"."""
    n = len(rows)
    sampled = [list(s) for s, _ in rows]
    num_tokens_no_spec = np.array([len(h) for _, h in rows], dtype=np.int64)
    token_ids_cpu = np.zeros((n, max_len), dtype=np.int64)
    for i, (_, h) in enumerate(rows):
        token_ids_cpu[i, :len(h)] = h
    return sampled, num_tokens_no_spec, token_ids_cpu


def _drafters(p):
    """row -> drafter_id of the sequence currently occupying it."""
    return {row: p._seqs[sid].drafter_id for row, sid in p._row_seq.items()}


@pytest.fixture(autouse=True)
def _reset_default_client():
    yield
    proposer.configure(None)


def test_multi_round_never_desyncs():
    gamma = 2
    client = InProcessDraftClient(EchoDraftEngine())
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client)

    prompt = [10, 11, 12]
    sampled, n_hist, tok = _batch([([1], prompt)])
    out = p.propose(sampled, n_hist, tok)
    assert len(out[0]) == gamma

    # Target accepted 1 of gamma and appended its own bonus token: history
    # grows by 2, and the new tokens are NOT what the drafter proposed.
    history2 = prompt + [77777, 88888]
    sampled, n_hist, tok = _batch([([1], history2)])
    out = p.propose(sampled, n_hist, tok)  # must not raise "desync"
    assert len(out[0]) == gamma

    history3 = history2 + [55555, 66666]
    sampled, n_hist, tok = _batch([([1], history3)])
    out = p.propose(sampled, n_hist, tok)
    assert len(out[0]) == gamma


def test_zero_accepted_round_is_still_consistent():
    """Full rejection: history grows by exactly 1 (the bonus token only)."""
    gamma = 3
    client = InProcessDraftClient(EchoDraftEngine())
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client)

    prompt = [1, 2, 3]
    sampled, n_hist, tok = _batch([([1], prompt)])
    p.propose(sampled, n_hist, tok)

    history2 = prompt + [999]
    sampled, n_hist, tok = _batch([([1], history2)])
    p.propose(sampled, n_hist, tok)  # must not raise


def test_slot_reuse_by_a_different_request_mints_a_fresh_seq_id():
    gamma = 2
    client = InProcessDraftClient(EchoDraftEngine())
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client)

    sampled, n_hist, tok = _batch([([1], [1, 2, 3, 4, 5])])
    p.propose(sampled, n_hist, tok)
    first_seq_id = p._row_seq[0]

    # Slot 0 handed to a shorter, unrelated request next step.
    sampled, n_hist, tok = _batch([([1], [9, 9])])
    p.propose(sampled, n_hist, tok)  # must not raise
    assert p._row_seq[0] != first_seq_id


def test_inactive_row_is_skipped_and_returns_empty():
    gamma = 2
    client = InProcessDraftClient(EchoDraftEngine())
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client)
    sampled, n_hist, tok = _batch([([], [1, 2, 3])])
    out = p.propose(sampled, n_hist, tok)
    assert out == [[]]


def test_uses_module_level_default_client():
    client = InProcessDraftClient(EchoDraftEngine())
    proposer.configure(client)
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(2))
    assert p.client is client


def test_raises_without_a_configured_client():
    with pytest.raises(RuntimeError, match="configure"):
        proposer.RemoteDraftProposer(_FakeVllmConfig(2))


def test_separate_instances_sharing_a_drafter_dont_collide_on_slot0():
    """Regression: a real drafter (VllmDraftEngine) outlives any single
    RemoteDraftProposer -- reused across target engine restarts. Two
    instances both minting "slot0-0" for their first request collided in
    the drafter's state dict and desynced on GPU (real bug, caught here)."""
    engine = EchoDraftEngine()
    client = InProcessDraftClient(engine)

    p1 = proposer.RemoteDraftProposer(_FakeVllmConfig(2), client=client)
    sampled, n_hist, tok = _batch([([1], [1, 2, 3])])
    p1.propose(sampled, n_hist, tok)

    p2 = proposer.RemoteDraftProposer(_FakeVllmConfig(2), client=client)
    sampled, n_hist, tok = _batch([([1], [9, 9, 9, 9, 9, 9])])
    p2.propose(sampled, n_hist, tok)  # must not raise "desync"

    assert p1._row_seq[0] != p2._row_seq[0]
    assert {p1._row_seq[0], p2._row_seq[0]} <= set(engine.state)


def test_no_routing_policy_keeps_every_request_on_the_fixed_drafter_id():
    """Backward-compat: omitting `routing` must reproduce the pre-fan-out
    behavior exactly, since every existing verify script and sbatch job
    relies on it."""
    gamma = 2
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client,
                                      drafter_id="d0")
    sampled, n_hist, tok = _batch([([1], [1, 2, 3]), ([1], [4, 5, 6])])
    p.propose(sampled, n_hist, tok)
    assert _drafters(p) == {0: "d0", 1: "d0"}


def test_routing_policy_assigns_fresh_requests_across_drafters():
    gamma = 2
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client,
                                      routing=RoundRobin())
    sampled, n_hist, tok = _batch([([1], [1, 2, 3]), ([1], [4, 5, 6])])
    p.propose(sampled, n_hist, tok)
    assert _drafters(p) == {0: "d0", 1: "d1"}


def test_routing_choice_is_sticky_for_the_life_of_the_request():
    """Picking a drafter only happens on the fresh round -- later rounds for
    the same slot must keep going to the same drafter, or that drafter's
    confirmed-length bookkeeping desyncs (it never saw the earlier rounds)."""
    gamma = 2
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client,
                                      routing=RoundRobin())

    sampled, n_hist, tok = _batch([([1], [1, 2, 3])])
    p.propose(sampled, n_hist, tok)
    chosen = _drafters(p)[0]

    for _ in range(3):
        history = [1, 2, 3] + [77, 88] * (_ + 1)
        sampled, n_hist, tok = _batch([([1], history)])
        p.propose(sampled, n_hist, tok)  # must not raise "desync"
        assert _drafters(p)[0] == chosen


def test_routing_picks_again_when_a_slot_is_handed_to_a_new_request():
    gamma = 2
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma), client=client,
                                      routing=RoundRobin())

    sampled, n_hist, tok = _batch([([1], [1, 2, 3])])
    p.propose(sampled, n_hist, tok)
    assert _drafters(p)[0] == "d0"

    # Slot 0 handed to a shorter, unrelated request next step -> fresh again.
    sampled, n_hist, tok = _batch([([1], [9, 9])])
    p.propose(sampled, n_hist, tok)
    assert _drafters(p)[0] == "d1"


def test_assignment_sink_records_only_routed_fresh_picks():
    gamma = 2
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    assignments: list[str] = []
    proposer.configure(client, routing=RoundRobin(), assignments=assignments)
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(gamma))

    sampled, n_hist, tok = _batch([([1], [1, 2, 3]), ([1], [4, 5, 6])])
    p.propose(sampled, n_hist, tok)
    assert assignments == ["d0", "d1"]

    # A non-fresh round for the same two slots must not append again.
    sampled, n_hist, tok = _batch([([1], [1, 2, 3, 7, 8]), ([1], [4, 5, 6, 9, 10])])
    p.propose(sampled, n_hist, tok)
    assert assignments == ["d0", "d1"]


def test_request_moved_to_another_row_keeps_its_identity():
    """InputBatch.condense()/swap_states() move requests between rows; the
    proposer must follow the content, not the row index."""
    client = InProcessDraftClient({"d0": EchoDraftEngine(node_id="d0"),
                                    "d1": EchoDraftEngine(node_id="d1")})
    rounds = []
    proposer.configure(client, routing=RoundRobin(), rounds=rounds)
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(2))
    a, b = [1, 2, 3], [4, 5, 6]
    p.propose(*_batch([([1], a), ([1], b)]))
    seq_a, seq_b = p._row_seq[0], p._row_seq[1]

    # swapped rows, both grew by 2
    p.propose(*_batch([([1], b + [7, 8]), ([1], a + [9, 9])]))  # must not desync
    assert p._row_seq == {0: seq_b, 1: seq_a}
    assert [r.prompt_token_ids is None for r in rounds[2:]] == [True, True]


def test_new_request_longer_than_previous_occupant_is_fresh():
    client = InProcessDraftClient(EchoDraftEngine())
    p = proposer.RemoteDraftProposer(_FakeVllmConfig(2), client=client)
    p.propose(*_batch([([1], [1, 2])]))
    first = p._row_seq[0]
    p.propose(*_batch([([1], [9] * 10)]))  # longer, unrelated: must be fresh
    assert p._row_seq[0] != first
