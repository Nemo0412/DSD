"""ZmqDraftClient / DraftServer over real loopback sockets. No GPU, no model.

Mirrors test_transport.py's InProcessDraftClient coverage so the two
transports are held to the same contract -- swapping one for the other must
not change what a caller can observe, only the wall-clock cost.
"""
from __future__ import annotations

import time

import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.draft.server import DraftServer
from dsdreal.netem.sleep import SleepEmulator
from dsdreal.proto.messages import SpeculateRequest
from dsdreal.proto.transport import ZmqDraftClient, make_client


def req(seq_id="s0", drafter="d0", gamma=4, accepted=(), n_conf=0, prompt=None):
    return SpeculateRequest(
        seq_id=seq_id, drafter_id=drafter, gamma=gamma,
        accepted_tokens=list(accepted), num_confirmed=n_conf,
        prompt_token_ids=prompt,
    )


@pytest.fixture
def one_drafter():
    server = DraftServer(EchoDraftEngine(node_id="d0"), "tcp://127.0.0.1:0").start()
    client = ZmqDraftClient({"d0": server.bound_addr}, timeout_s=5.0)
    yield client, server
    client.close()
    server.stop()


def test_round_trip_matches_inprocess_semantics(one_drafter):
    client, _ = one_drafter
    (reply,) = client.speculate([req(gamma=4, prompt=[1, 2, 3])])
    assert len(reply.draft_tokens) == 4


def test_replies_preserve_request_order(one_drafter):
    client, _ = one_drafter
    reqs = [req(seq_id=f"s{i}", gamma=i % 3 + 1, prompt=[1, 2, 3]) for i in range(8)]
    replies = client.speculate(reqs)
    assert [r.seq_id for r in replies] == [r.seq_id for r in reqs]
    assert [len(r.draft_tokens) for r in replies] == [r.gamma for r in reqs]


def test_made_via_factory():
    server = DraftServer(EchoDraftEngine(node_id="d0"), "tcp://127.0.0.1:0").start()
    client = make_client("zmq", addresses={"d0": server.bound_addr})
    try:
        (reply,) = client.speculate([req(prompt=[1])])
        assert reply.draft_tokens
    finally:
        client.close()
        server.stop()


def test_free_releases_state(one_drafter):
    client, server = one_drafter
    client.speculate([req(seq_id="s1", prompt=[1]), req(seq_id="s2", prompt=[1])])
    # free() is fire-and-forget; give the server thread a moment to apply it.
    client.free(["s1"])
    deadline = time.perf_counter() + 2.0
    while "s1" in server.engine.state and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert set(server.engine.state) == {"s2"}


def test_unknown_drafter_raises_before_touching_the_network(one_drafter):
    client, _ = one_drafter
    with pytest.raises(KeyError, match="d9"):
        client.speculate([req(drafter="d9", prompt=[1])])


def test_engine_desync_becomes_a_client_side_valueerror(one_drafter):
    client, _ = one_drafter
    client.speculate([req(gamma=4, n_conf=3, prompt=[10, 11, 12])])
    with pytest.raises(ValueError, match="desync"):
        client.speculate([req(gamma=4, accepted=[7], n_conf=99)])


def test_unreachable_drafter_times_out():
    # Nothing bound on this port: DEALER connects happily, nothing ever replies.
    client = ZmqDraftClient({"d0": "tcp://127.0.0.1:1"}, timeout_s=0.3)
    try:
        with pytest.raises(TimeoutError):
            client.speculate([req(prompt=[1])])
    finally:
        client.close()


def test_fanout_across_drafters_costs_max_not_sum():
    """Two drafters, each with real per-token compute delay, on real threads
    and real sockets. Wall clock must track the slower one, not their sum --
    the whole point of sending everything before receiving anything."""
    compute_ms = 40.0
    servers = [
        DraftServer(EchoDraftEngine(node_id=f"d{i}", compute_ms_per_token=compute_ms / 4),
                    "tcp://127.0.0.1:0").start()
        for i in range(2)
    ]
    client = ZmqDraftClient({f"d{i}": s.bound_addr for i, s in enumerate(servers)})
    try:
        t0 = time.perf_counter()
        client.speculate([req(seq_id=f"s{i}", drafter=f"d{i}", gamma=4, prompt=[1])
                           for i in range(2)])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert wall_ms < compute_ms * 1.5, f"fan-out serialised: {wall_ms:.2f}ms"
    finally:
        client.close()
        for s in servers:
            s.stop()


def test_injected_rtt_shows_up_in_the_round_trip():
    server = DraftServer(EchoDraftEngine(node_id="d0"), "tcp://127.0.0.1:0").start()
    rtt_ms = 4.0
    client = ZmqDraftClient({"d0": server.bound_addr},
                             emulator=SleepEmulator(rtt_ms=rtt_ms))
    try:
        t0 = time.perf_counter()
        (reply,) = client.speculate([req(prompt=[1])])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert reply.injected_delay_ms == pytest.approx(rtt_ms, abs=1e-6)
        assert wall_ms >= rtt_ms * 0.9
        # Real sockets + a server thread cost more overhead than the
        # in-process path's <1ms bound; loopback should still stay well
        # under it.
        assert reply.net_excess_ms < 20.0
    finally:
        client.close()
        server.stop()


def test_empty_batch_is_free(one_drafter):
    client, _ = one_drafter
    t0 = time.perf_counter()
    assert client.speculate([]) == []
    assert (time.perf_counter() - t0) * 1000.0 < 5.0


def test_drafter_ids_lists_the_registry():
    servers = [DraftServer(EchoDraftEngine(node_id=f"d{i}"), "tcp://127.0.0.1:0").start()
               for i in range(3)]
    client = ZmqDraftClient({f"d{i}": s.bound_addr for i, s in enumerate(servers)})
    try:
        assert sorted(client.drafter_ids) == ["d0", "d1", "d2"]
    finally:
        client.close()
        for s in servers:
            s.stop()
