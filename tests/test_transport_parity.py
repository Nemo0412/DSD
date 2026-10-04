"""The three transports must be indistinguishable above DraftClient.

test_transport.py, test_zmq_transport.py and test_http_transport.py each
assert the contract separately, which catches a transport that is wrong on
its own terms. This file catches the other failure: two transports that are
each self-consistent but disagree with each other -- a drift that would make
"fused vs distributed" and "ZMQ vs HTTP" comparisons meaningless, because
the transport would be changing the tokens rather than only their cost.

Everything here is no-GPU: EchoDraftEngine's tokens are a pure function of
(seq_id, position), so any divergence is the transport's fault.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.draft.http_server import HttpDraftServer
from dsdreal.draft.server import DraftServer
from dsdreal.proto.messages import SpeculateRequest
from dsdreal.proto.transport import (
    HttpDraftClient,
    InProcessDraftClient,
    ZmqDraftClient,
)

DRAFTERS = ("d0", "d1")


@contextmanager
def inprocess_client():
    engines = {d: EchoDraftEngine(node_id=d) for d in DRAFTERS}
    client = InProcessDraftClient(engines)
    try:
        yield client
    finally:
        client.close()


@contextmanager
def zmq_client():
    servers = {d: DraftServer(EchoDraftEngine(node_id=d),
                              "tcp://127.0.0.1:0").start() for d in DRAFTERS}
    client = ZmqDraftClient({d: s.bound_addr for d, s in servers.items()})
    try:
        yield client
    finally:
        client.close()
        for s in servers.values():
            s.stop()


@contextmanager
def http_client():
    servers = {d: HttpDraftServer(EchoDraftEngine(node_id=d)).start()
               for d in DRAFTERS}
    client = HttpDraftClient({d: s.base_url for d, s in servers.items()})
    try:
        yield client
    finally:
        client.close()
        for s in servers.values():
            s.stop()


TRANSPORTS = {"inprocess": inprocess_client, "zmq": zmq_client, "http": http_client}


def _run_session(client) -> list[tuple[str, tuple[int, ...]]]:
    """A few rounds of the real propose/rollback loop, two drafters in each
    batch, variable gamma, partial acceptance with a bonus token."""
    prompts = {"s0": [10, 11, 12], "s1": [20, 21]}
    confirmed = {sid: list(p) for sid, p in prompts.items()}
    out: list[tuple[str, tuple[int, ...]]] = []

    batch = [
        SpeculateRequest(seq_id="s0", drafter_id="d0", gamma=4,
                         num_confirmed=3, prompt_token_ids=prompts["s0"]),
        SpeculateRequest(seq_id="s1", drafter_id="d1", gamma=2,
                         num_confirmed=2, prompt_token_ids=prompts["s1"]),
    ]
    for _ in range(3):
        replies = client.speculate(batch)
        out += [(r.seq_id, tuple(r.draft_tokens)) for r in replies]

        next_batch = []
        for request, reply in zip(batch, replies):
            # Accept half, then the target's own bonus/correction token --
            # the case where confirmed is NOT a prefix of what was proposed.
            accepted = list(reply.draft_tokens[: request.gamma // 2]) + [99_000]
            confirmed[request.seq_id] += accepted
            next_batch.append(SpeculateRequest(
                seq_id=request.seq_id, drafter_id=request.drafter_id,
                gamma=request.gamma, accepted_tokens=accepted,
                num_confirmed=len(confirmed[request.seq_id]),
            ))
        batch = next_batch

    client.free(["s0"])
    return out


@pytest.mark.parametrize("name", ["zmq", "http"])
def test_remote_transport_matches_inprocess_token_for_token(name):
    with inprocess_client() as client:
        expected = _run_session(client)
    with TRANSPORTS[name]() as client:
        assert _run_session(client) == expected


def test_zmq_and_http_agree():
    """Explicit rather than implied by transitivity: these two are the pair
    an experiment actually chooses between."""
    with zmq_client() as client:
        via_zmq = _run_session(client)
    with http_client() as client:
        via_http = _run_session(client)
    assert via_zmq == via_http


@pytest.mark.parametrize("name", list(TRANSPORTS))
def test_every_transport_reports_the_same_registry(name):
    with TRANSPORTS[name]() as client:
        assert sorted(client.drafter_ids) == list(DRAFTERS)
