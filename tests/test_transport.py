"""Transport-layer tests. No GPU, no model, no sockets."""
from __future__ import annotations

import time

import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.netem.null import NullEmulator
from dsdreal.netem.sleep import SleepEmulator
from dsdreal.proto.messages import SpeculateRequest
from dsdreal.proto.transport import InProcessDraftClient, make_client


def req(seq_id="s0", drafter="d0", gamma=4, accepted=(), n_conf=0, prompt=None):
    return SpeculateRequest(
        seq_id=seq_id, drafter_id=drafter, gamma=gamma,
        accepted_tokens=list(accepted), num_confirmed=n_conf,
        prompt_token_ids=prompt,
    )


def test_replies_preserve_request_order():
    engines = {f"d{i}": EchoDraftEngine(node_id=f"d{i}") for i in range(4)}
    client = InProcessDraftClient(engines)
    reqs = [req(seq_id=f"s{i}", drafter=f"d{i % 4}", gamma=i % 3 + 1,
                prompt=[1, 2, 3]) for i in range(8)]
    replies = client.speculate(reqs)
    assert [r.seq_id for r in replies] == [r.seq_id for r in reqs]
    assert [len(r.draft_tokens) for r in replies] == [r.gamma for r in reqs]


def test_variable_gamma_within_one_batch():
    client = InProcessDraftClient(EchoDraftEngine())
    reqs = [req(seq_id=f"s{i}", gamma=g, prompt=[7])
            for i, g in enumerate([1, 2, 3, 4, 8])]
    replies = client.speculate(reqs)
    assert [len(r.draft_tokens) for r in replies] == [1, 2, 3, 4, 8]


def test_gamma_zero_yields_no_tokens():
    client = InProcessDraftClient(EchoDraftEngine())
    (reply,) = client.speculate([req(gamma=0, prompt=[7])])
    assert reply.draft_tokens == []


def test_rollback_drops_the_speculative_tail():
    """State must be a function of the confirmed prefix alone.

    After proposing gamma tokens the drafter sits gamma ahead. When the target
    confirms fewer, that tail is dropped — and what it proposes next must
    depend only on what was confirmed, never on what was rejected.
    """
    engine = EchoDraftEngine()
    client = InProcessDraftClient(engine)
    prompt = [10, 11, 12]

    (first,) = client.speculate([req(gamma=4, n_conf=3, prompt=prompt)])
    assert engine.state["s0"].speculative == 4

    # The target accepted 2 of 4 and appended its own correction token, so the
    # confirmed tokens are NOT a prefix of what we proposed.
    accepted = first.draft_tokens[:2] + [99999]
    (second,) = client.speculate([req(gamma=4, accepted=accepted, n_conf=6)])

    assert engine.rollbacks == 1
    assert engine.rolled_back_tokens == 4
    assert engine.state["s0"].confirmed == prompt + accepted

    # A drafter built fresh at the same confirmed prefix must agree.
    fresh = EchoDraftEngine()
    (mirror,) = InProcessDraftClient(fresh).speculate(
        [req(gamma=4, n_conf=6, prompt=prompt + accepted)])
    assert second.draft_tokens == mirror.draft_tokens


def test_desync_fails_loudly():
    """A drafter that disagrees with the target about confirmed length must
    raise, not silently draft from the wrong prefix."""
    client = InProcessDraftClient(EchoDraftEngine())
    client.speculate([req(gamma=4, n_conf=3, prompt=[1, 2, 3])])
    with pytest.raises(ValueError, match="desync"):
        client.speculate([req(gamma=4, accepted=[7], n_conf=99)])


def test_free_releases_state():
    engine = EchoDraftEngine()
    client = InProcessDraftClient(engine)
    client.speculate([req(seq_id="s1", prompt=[1]), req(seq_id="s2", prompt=[1])])
    assert set(engine.state) == {"s1", "s2"}
    client.free(["s1"])
    assert set(engine.state) == {"s2"}


def test_unknown_drafter_names_the_registry():
    client = InProcessDraftClient(EchoDraftEngine(node_id="d0"))
    with pytest.raises(KeyError, match="d9"):
        client.speculate([req(drafter="d9", prompt=[1])])


@pytest.mark.parametrize("rtt_ms", [0.0, 1.0, 4.0])
def test_injected_rtt_shows_up_in_the_round_trip(rtt_ms):
    client = InProcessDraftClient(
        EchoDraftEngine(), emulator=SleepEmulator(rtt_ms=rtt_ms))
    t0 = time.perf_counter()
    (reply,) = client.speculate([req(prompt=[1])])
    wall_ms = (time.perf_counter() - t0) * 1000.0

    assert reply.injected_delay_ms == pytest.approx(rtt_ms, abs=1e-6)
    assert wall_ms >= rtt_ms * 0.9
    # Everything above the injected delay and drafter compute is transport
    # overhead the simulator does not model; on this path it must be tiny.
    assert reply.net_excess_ms < 1.0


def test_fanout_costs_max_rtt_not_sum():
    """N drafters in one batch must not serialise their delays."""
    n = 6
    rtt = 3.0
    engines = {f"d{i}": EchoDraftEngine(node_id=f"d{i}") for i in range(n)}
    client = InProcessDraftClient(engines, emulator=SleepEmulator(rtt_ms=rtt))
    reqs = [req(seq_id=f"s{i}", drafter=f"d{i}", prompt=[1]) for i in range(n)]

    t0 = time.perf_counter()
    client.speculate(reqs)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    assert wall_ms < rtt * 2, f"fan-out serialised: {wall_ms:.2f}ms for n={n}"


def test_empty_batch_is_free():
    client = InProcessDraftClient(EchoDraftEngine(), emulator=SleepEmulator(rtt_ms=50))
    t0 = time.perf_counter()
    assert client.speculate([]) == []
    assert (time.perf_counter() - t0) * 1000.0 < 5.0


def test_jitter_none_is_deterministic():
    e = SleepEmulator(rtt_ms=2.0, jitter_ms=0.0)
    assert {e.one_way_ms("t0", "d0") for _ in range(10)} == {1.0}


def test_latency_matrix_overrides_default_and_is_symmetric():
    e = SleepEmulator(rtt_ms=1.0, matrix={("t0", "d1"): 10.0})
    assert e.one_way_ms("t0", "d1") == 5.0
    assert e.one_way_ms("d1", "t0") == 5.0
    assert e.one_way_ms("t0", "d2") == 0.5


def test_null_emulator_injects_nothing():
    assert NullEmulator().one_way_ms("a", "b") == 0.0


def test_factory_rejects_unknown_transport():
    with pytest.raises(ValueError, match="unknown transport"):
        make_client("carrier-pigeon")


def test_drafter_ids_lists_the_registry():
    engines = {f"d{i}": EchoDraftEngine(node_id=f"d{i}") for i in range(3)}
    client = InProcessDraftClient(engines)
    assert sorted(client.drafter_ids) == ["d0", "d1", "d2"]
