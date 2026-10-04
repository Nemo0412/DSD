"""How a target reaches its drafters.

`propose()` runs synchronously inside the vLLM engine step, so this API is
blocking by construction — no asyncio anywhere on this path.

The batch contract matters: `speculate()` takes the whole step's requests at
once so an implementation can fan out concurrently. Wall clock must be
max(RTT) over drafters, never the sum; a per-request API would silently
serialise N draft nodes and destroy the thing we are trying to measure.

Three implementations, one contract: InProcessDraftClient (no sockets, for
fused mode and every no-GPU test), ZmqDraftClient (DEALER/ROUTER + msgpack)
and HttpDraftClient (HTTP/JSON). The two remote ones are kept deliberately
interchangeable so transport choice stays an experiment rather than an
assumption -- `SpeculateReply.net_excess_ms` is what decides between them.
"""
from __future__ import annotations

import http.client
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import msgspec
import zmq

from dsdreal.draft.base import speculate_many
from dsdreal.netem.base import NetworkEmulator, block_for
from dsdreal.netem.null import NullEmulator
from dsdreal.proto.messages import FreeSeq, SpeculateReply, SpeculateRequest


class DraftClient(ABC):
    """Target-side handle on one or more drafters."""

    def __init__(self, node_id: str = "t0",
                 emulator: NetworkEmulator | None = None) -> None:
        self.node_id = node_id
        self.emulator = emulator or NullEmulator()

    @abstractmethod
    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        """Replies in the same order as `requests`."""

    @abstractmethod
    def free(self, seq_ids: Iterable[str]) -> None: ...

    @property
    @abstractmethod
    def drafter_ids(self) -> list[str]:
        """Every drafter this client can reach, for a RoutingPolicy to pick from."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class InProcessDraftClient(DraftClient):
    """Simplest possible transport: call the drafter directly.

    No sockets, no serialization. Used by Stage 2 (fused mode) and by every
    no-GPU test. Swapping this for ZmqDraftClient changes nothing above
    this interface — that swap is real now, not a placeholder.

    Delay is still applied, so fused runs produce comparable timings. Because
    the calls are inevitably serial here, the emulated delay is charged once
    per direction as max() over the batch, mirroring what a concurrent
    fan-out would cost rather than what a serial loop would.
    """

    def __init__(self, engines, node_id: str = "t0", emulator=None) -> None:
        super().__init__(node_id=node_id, emulator=emulator)
        if not isinstance(engines, dict):
            engines = {engines.node_id: engines}
        self.engines = engines

    def _engine_for(self, drafter_id: str):
        try:
            return self.engines[drafter_id]
        except KeyError:
            raise KeyError(
                f"unknown drafter_id {drafter_id!r}; "
                f"registered: {sorted(self.engines)}"
            ) from None

    @property
    def drafter_ids(self) -> list[str]:
        return list(self.engines)

    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        if not requests:
            return []
        sent = time.perf_counter()
        out_ms = [self.emulator.one_way_ms(self.node_id, r.drafter_id)
                  for r in requests]
        block_for(max(out_ms) / 1000.0)

        # Grouped per drafter so a batching engine (VllmDraftEngine) serves
        # its share of the step in one forward, as it would behind a server.
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault(r.drafter_id, []).append(i)
        replies: list[SpeculateReply] = [None] * len(requests)  # type: ignore[list-item]
        for drafter_id, idx in groups.items():
            results = speculate_many(self._engine_for(drafter_id), [requests[i] for i in idx])
            for i, res in zip(idx, results):
                if isinstance(res, Exception):
                    raise res
                replies[i] = res

        back_ms = [self.emulator.one_way_ms(r.drafter_id, self.node_id)
                   for r in requests]
        block_for(max(back_ms) / 1000.0)
        recv = time.perf_counter()

        for reply, o, b in zip(replies, out_ms, back_ms):
            reply.sent_ms = sent * 1000.0
            reply.recv_ms = recv * 1000.0
            reply.injected_delay_ms = o + b
        return replies

    def free(self, seq_ids: Iterable[str]) -> None:
        ids = list(seq_ids)
        for engine in self.engines.values():
            engine.free(ids)

    def close(self) -> None:
        for engine in self.engines.values():
            engine.close()


class ZmqDraftClient(DraftClient):
    """One DEALER socket per drafter, matching InProcessDraftClient's dict-
    keyed-by-drafter_id shape so swapping transports changes nothing above
    this interface.

    A batch fans out concurrently: every request is sent before any recv, so
    wall clock is bounded by the slowest drafter's RTT + compute, not their
    sum. See draft/server.py for the wire framing this speaks.
    """

    def __init__(self, addresses: dict[str, str], node_id: str = "t0",
                 emulator: NetworkEmulator | None = None,
                 context: "zmq.Context | None" = None,
                 timeout_s: float = 5.0) -> None:
        super().__init__(node_id=node_id, emulator=emulator)
        self._ctx = context or zmq.Context.instance()
        self._sockets: dict[str, zmq.Socket] = {}
        for drafter_id, addr in addresses.items():
            sock = self._ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(addr)
            self._sockets[drafter_id] = sock
        self._timeout_s = timeout_s
        self._enc = msgspec.msgpack.Encoder()
        self._reply_dec = msgspec.msgpack.Decoder(SpeculateReply)

    def _socket_for(self, drafter_id: str) -> "zmq.Socket":
        try:
            return self._sockets[drafter_id]
        except KeyError:
            raise KeyError(
                f"unknown drafter_id {drafter_id!r}; "
                f"registered: {sorted(self._sockets)}"
            ) from None

    @property
    def drafter_ids(self) -> list[str]:
        return list(self._sockets)

    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        if not requests:
            return []
        sent = time.perf_counter()
        out_ms = [self.emulator.one_way_ms(self.node_id, r.drafter_id)
                  for r in requests]
        block_for(max(out_ms) / 1000.0)

        sockets_used = set()
        for r in requests:
            sock = self._socket_for(r.drafter_id)
            sockets_used.add(sock)
            sock.send_multipart([b"SPEC", r.seq_id.encode(), self._enc.encode(r)])

        poller = zmq.Poller()
        for sock in sockets_used:
            poller.register(sock, zmq.POLLIN)

        pending = {r.seq_id for r in requests}
        replies: dict[str, SpeculateReply] = {}
        deadline = time.perf_counter() + self._timeout_s
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"no reply from drafters for {sorted(pending)}")
            for sock, ev in poller.poll(timeout=remaining * 1000.0):
                if ev != zmq.POLLIN:
                    continue
                msg_type, corr_id, payload = sock.recv_multipart()
                seq_id = corr_id.decode()
                if msg_type == b"ERR ":
                    raise ValueError(f"drafter error on {seq_id}: {payload.decode()}")
                replies[seq_id] = self._reply_dec.decode(payload)
                pending.discard(seq_id)

        back_ms = [self.emulator.one_way_ms(r.drafter_id, self.node_id)
                   for r in requests]
        block_for(max(back_ms) / 1000.0)
        recv = time.perf_counter()

        out = []
        for r, o, b in zip(requests, out_ms, back_ms):
            reply = replies[r.seq_id]
            reply.sent_ms = sent * 1000.0
            reply.recv_ms = recv * 1000.0
            reply.injected_delay_ms = o + b
            out.append(reply)
        return out

    def free(self, seq_ids: Iterable[str]) -> None:
        payload = self._enc.encode(FreeSeq(seq_ids=list(seq_ids)))
        for sock in self._sockets.values():
            sock.send_multipart([b"FREE", b"", payload])

    def close(self) -> None:
        for sock in self._sockets.values():
            sock.close(linger=0)


class HttpDraftClient(DraftClient):
    """HTTP/JSON sibling of ZmqDraftClient. Same dict-keyed-by-drafter_id
    shape, same batch contract, inspectable wire.

    Concurrency is structural rather than pipelined: a batch is grouped by
    drafter and each group goes out as ONE POST carrying that drafter's whole
    sublist, all POSTs in flight at once on a thread per drafter. So wall
    clock is max over drafters of (RTT + that drafter's compute), and the
    number of round trips is the number of drafters -- not the batch size.
    HTTP/1.1 cannot pipeline, so a POST-per-request client would serialise
    every sequence on the same connection instead. See draft/http_server.py.

    One persistent connection per drafter, reused across steps: http.client
    sets TCP_NODELAY on connect, and the server keeps the connection alive,
    so a step costs one round trip and no handshake.
    """

    def __init__(self, addresses: dict[str, str], node_id: str = "t0",
                 emulator: NetworkEmulator | None = None,
                 timeout_s: float = 5.0) -> None:
        super().__init__(node_id=node_id, emulator=emulator)
        self._addresses = dict(addresses)
        self._timeout_s = timeout_s
        self._conns: dict[str, http.client.HTTPConnection] = {}
        self._lock = threading.Lock()
        self._enc = msgspec.json.Encoder()
        self._reply_dec = msgspec.json.Decoder(list[SpeculateReply])
        self._err_dec = msgspec.json.Decoder(dict)
        self._pool = ThreadPoolExecutor(
            max_workers=max(len(self._addresses), 1),
            thread_name_prefix="dsd-http")

    @property
    def drafter_ids(self) -> list[str]:
        return list(self._addresses)

    def _url_for(self, drafter_id: str) -> str:
        try:
            return self._addresses[drafter_id]
        except KeyError:
            raise KeyError(
                f"unknown drafter_id {drafter_id!r}; "
                f"registered: {sorted(self._addresses)}"
            ) from None

    def _conn_for(self, drafter_id: str) -> "http.client.HTTPConnection":
        with self._lock:
            conn = self._conns.get(drafter_id)
            if conn is None:
                url = urlparse(self._url_for(drafter_id))
                conn = http.client.HTTPConnection(
                    url.hostname, url.port, timeout=self._timeout_s)
                self._conns[drafter_id] = conn
            return conn

    def _post(self, drafter_id: str, path: str, body: bytes,
              expect_reply: bool) -> bytes | None:
        """One POST on the drafter's persistent connection.

        Retried once on a dropped keep-alive: the server may have closed an
        idle connection between steps, which surfaces as an exception only
        when we next write. A retry is safe because `num_confirmed` makes
        speculate idempotent -- the drafter rolls back to an absolute length
        rather than applying a delta.
        """
        conn = self._conn_for(drafter_id)
        for attempt in (0, 1):
            try:
                conn.request("POST", path, body,
                             {"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = response.read()
                break
            except (http.client.HTTPException, OSError):
                conn.close()
                if attempt:
                    raise
        else:  # pragma: no cover - the loop always breaks or raises
            raise AssertionError

        if response.status == 400:
            message = self._err_dec.decode(payload).get("error", payload.decode())
            raise ValueError(f"drafter {drafter_id} error: {message}")
        if response.status >= 400:
            raise RuntimeError(
                f"drafter {drafter_id} returned {response.status}: {payload.decode()}")
        return payload if expect_reply else None

    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        if not requests:
            return []
        grouped: dict[str, list[SpeculateRequest]] = {}
        for r in requests:
            self._url_for(r.drafter_id)  # raise before touching the network
            grouped.setdefault(r.drafter_id, []).append(r)

        sent = time.perf_counter()
        out_ms = [self.emulator.one_way_ms(self.node_id, r.drafter_id)
                  for r in requests]
        block_for(max(out_ms) / 1000.0)

        futures = {
            drafter_id: self._pool.submit(
                self._post, drafter_id, "/speculate", self._enc.encode(group), True)
            for drafter_id, group in grouped.items()
        }
        replies: dict[str, SpeculateReply] = {}
        for future in futures.values():
            for reply in self._reply_dec.decode(future.result(self._timeout_s)):
                replies[reply.seq_id] = reply

        back_ms = [self.emulator.one_way_ms(r.drafter_id, self.node_id)
                   for r in requests]
        block_for(max(back_ms) / 1000.0)
        recv = time.perf_counter()

        out = []
        for r, o, b in zip(requests, out_ms, back_ms):
            reply = replies[r.seq_id]
            reply.sent_ms = sent * 1000.0
            reply.recv_ms = recv * 1000.0
            reply.injected_delay_ms = o + b
            out.append(reply)
        return out

    def free(self, seq_ids: Iterable[str]) -> None:
        body = self._enc.encode(FreeSeq(seq_ids=list(seq_ids)))
        # Unlike ZMQ's fire-and-forget FREE frame, HTTP always answers; the
        # 204 is waited on per drafter but the drafters go in parallel.
        futures = [self._pool.submit(self._post, d, "/free", body, False)
                   for d in self._addresses]
        for future in futures:
            future.result(self._timeout_s)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        with self._lock:
            for conn in self._conns.values():
                conn.close()
            self._conns.clear()


def make_client(kind: str, **kwargs) -> DraftClient:
    """Factory so config can name a transport without importing it."""
    if kind in ("inprocess", "local", "fused"):
        return InProcessDraftClient(**kwargs)
    if kind == "zmq":
        return ZmqDraftClient(**kwargs)
    if kind == "http":
        return HttpDraftClient(**kwargs)
    raise ValueError(f"unknown transport {kind!r}")
