"""HttpDraftClient / HttpDraftServer over real loopback sockets. No GPU.

Deliberately a near-copy of test_zmq_transport.py: the two remote transports
must be indistinguishable from above the DraftClient interface, so they are
held to the same assertions. Where this file diverges from the ZMQ one, the
divergence is the point and is commented.
"""
from __future__ import annotations

import time

import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.draft.http_server import HttpDraftServer
from dsdreal.netem.sleep import SleepEmulator
from dsdreal.proto.messages import SpeculateRequest
from dsdreal.proto.transport import HttpDraftClient, make_client


def req(seq_id="s0", drafter="d0", gamma=4, accepted=(), n_conf=0, prompt=None):
    return SpeculateRequest(
        seq_id=seq_id, drafter_id=drafter, gamma=gamma,
        accepted_tokens=list(accepted), num_confirmed=n_conf,
        prompt_token_ids=prompt,
    )


@pytest.fixture
def one_drafter():
    server = HttpDraftServer(EchoDraftEngine(node_id="d0")).start()
    client = HttpDraftClient({"d0": server.base_url}, timeout_s=5.0)
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
    server = HttpDraftServer(EchoDraftEngine(node_id="d0")).start()
    client = make_client("http", addresses={"d0": server.base_url})
    try:
        (reply,) = client.speculate([req(prompt=[1])])
        assert reply.draft_tokens
    finally:
        client.close()
        server.stop()


def test_free_releases_state(one_drafter):
    client, server = one_drafter
    client.speculate([req(seq_id="s1", prompt=[1]), req(seq_id="s2", prompt=[1])])
    # Unlike ZMQ's fire-and-forget FREE, HTTP answers 204, so by the time
    # free() returns the drafter has already applied it -- no polling.
    client.free(["s1"])
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


def test_drafter_survives_an_error_and_keeps_serving(one_drafter):
    """A 400 must not wedge the connection: the next step has to work.

    HTTP-specific. The ZMQ side gets this for free (datagram framing); here a
    mid-stream error still has to leave a reusable keep-alive connection.
    """
    client, _ = one_drafter
    client.speculate([req(seq_id="a", gamma=2, n_conf=3, prompt=[1, 2, 3])])
    with pytest.raises(ValueError, match="desync"):
        client.speculate([req(seq_id="a", gamma=2, accepted=[7], n_conf=99)])
    (reply,) = client.speculate([req(seq_id="b", gamma=3, prompt=[4, 5])])
    assert len(reply.draft_tokens) == 3


def test_unreachable_drafter_raises():
    # Nothing listening on this port. Unlike ZMQ's DEALER (which connects
    # optimistically and then waits out the timeout), TCP refuses the
    # connection outright, so this surfaces as OSError rather than TimeoutError.
    client = HttpDraftClient({"d0": "http://127.0.0.1:1"}, timeout_s=0.3)
    try:
        with pytest.raises((OSError, TimeoutError)):
            client.speculate([req(prompt=[1])])
    finally:
        client.close()


def test_fanout_across_drafters_costs_max_not_sum():
    """Two drafters, real per-token compute, real sockets and threads.

    This is the assertion that would fail if the client POSTed one request at
    a time on a shared connection.
    """
    compute_ms = 40.0
    servers = [
        HttpDraftServer(EchoDraftEngine(node_id=f"d{i}",
                                        compute_ms_per_token=compute_ms / 4)).start()
        for i in range(2)
    ]
    client = HttpDraftClient({f"d{i}": s.base_url for i, s in enumerate(servers)})
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


def test_one_round_trip_per_drafter_regardless_of_batch_size():
    """HTTP-specific: many sequences on ONE drafter must still be one POST.

    Counted server-side rather than timed. A timing assertion cannot see
    this on loopback -- eight extra round trips there cost well under a
    millisecond -- so a per-request-POST regression would slip straight
    through it, and only show up over a real link.
    """
    server = HttpDraftServer(EchoDraftEngine(node_id="d0")).start()
    client = HttpDraftClient({"d0": server.base_url})
    try:
        client.speculate([req(seq_id=f"s{i}", gamma=2, prompt=[1]) for i in range(8)])
        assert server.speculate_posts == 1
        client.speculate([req(seq_id=f"s{i}", gamma=2, accepted=[1, 2, 3], n_conf=4)
                          for i in range(8)])
        assert server.speculate_posts == 2
    finally:
        client.close()
        server.stop()


def test_two_drafters_cost_one_round_trip_each():
    servers = [HttpDraftServer(EchoDraftEngine(node_id=f"d{i}")).start()
               for i in range(2)]
    client = HttpDraftClient({f"d{i}": s.base_url for i, s in enumerate(servers)})
    try:
        client.speculate([req(seq_id=f"s{i}", drafter=f"d{i % 2}", prompt=[1])
                          for i in range(6)])
        assert [s.speculate_posts for s in servers] == [1, 1]
    finally:
        client.close()
        for s in servers:
            s.stop()


def test_injected_rtt_shows_up_in_the_round_trip():
    server = HttpDraftServer(EchoDraftEngine(node_id="d0")).start()
    rtt_ms = 4.0
    client = HttpDraftClient({"d0": server.base_url},
                             emulator=SleepEmulator(rtt_ms=rtt_ms))
    try:
        t0 = time.perf_counter()
        (reply,) = client.speculate([req(prompt=[1])])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert reply.injected_delay_ms == pytest.approx(rtt_ms, abs=1e-6)
        assert wall_ms >= rtt_ms * 0.9
        # Same bound as the ZMQ path. If JSON + HTTP framing ever costs
        # materially more than msgpack + ZMQ on loopback, this is where it
        # shows up first -- and net_excess_ms is the number to compare.
        assert reply.net_excess_ms < 20.0
    finally:
        client.close()
        server.stop()


def test_keepalive_is_reused_across_steps(one_drafter):
    """Each step must not pay a TCP handshake.

    Indirect but sufficient: the client holds exactly one connection object
    for the drafter across many steps, and it is never reopened.
    """
    client, _ = one_drafter
    client.speculate([req(seq_id="s0", prompt=[1])])
    conn = client._conns["d0"]
    for i in range(1, 5):
        client.speculate([req(seq_id=f"s{i}", prompt=[1])])
    assert client._conns["d0"] is conn
    assert conn.sock is not None, "connection was closed between steps"


def test_empty_batch_is_free(one_drafter):
    client, _ = one_drafter
    t0 = time.perf_counter()
    assert client.speculate([]) == []
    assert (time.perf_counter() - t0) * 1000.0 < 5.0


def test_drafter_ids_lists_the_registry():
    servers = [HttpDraftServer(EchoDraftEngine(node_id=f"d{i}")).start()
               for i in range(3)]
    client = HttpDraftClient({f"d{i}": s.base_url for i, s in enumerate(servers)})
    try:
        assert sorted(client.drafter_ids) == ["d0", "d1", "d2"]
    finally:
        client.close()
        for s in servers:
            s.stop()
