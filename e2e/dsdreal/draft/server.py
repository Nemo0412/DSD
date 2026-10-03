"""ZMQ ROUTER server: runs a DraftEngine so a target reaches it over a real
socket instead of calling it in-process.

Wire framing between this and proto/transport.py's ZmqDraftClient (the
DEALER side) is 3 frames after ROUTER's auto-prepended identity:

    [msg_type, corr_id, payload]

msg_type is b"SPEC" (payload is a msgpack'd SpeculateRequest, corr_id is its
seq_id, a reply is expected), b"FREE" (payload is a msgpack'd FreeSeq,
corr_id unused, fire-and-forget -- cleanup isn't on vLLM's synchronous
critical path the way propose() is, so there is nothing worth blocking on),
or b"ERR " (server->client only: the engine raised on this seq_id: payload
is the error message, utf-8). EchoDraftEngine's desync check is the reason
this exists -- across a socket that must become a reply the client re-raises
rather than a thread that quietly dies and leaves the client waiting out its
timeout.
"""
from __future__ import annotations

import os
import threading
import time

import msgspec
import zmq

from dsdreal.draft.base import DraftEngine, speculate_many
from dsdreal.proto.messages import FreeSeq, SpeculateRequest

_SPEC = b"SPEC"
_FREE = b"FREE"
_ERR = b"ERR "
_LOG = os.environ.get("DSD_DRAFT_LOG") == "1"


class DraftServer:
    """One process per drafter node. `bind_addr` is what a ZmqDraftClient's
    `addresses[engine.node_id]` must point at -- use `self.bound_addr` after
    `start()` if `bind_addr` used an ephemeral port (":0")."""

    def __init__(self, engine: DraftEngine, bind_addr: str, context=None) -> None:
        self.engine = engine
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(bind_addr)
        self.bound_addr = self._sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._req_dec = msgspec.msgpack.Decoder(SpeculateRequest)
        self._free_dec = msgspec.msgpack.Decoder(FreeSeq)
        self._enc = msgspec.msgpack.Encoder()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "DraftServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=100)):
                continue
            # A client sends one step's requests back to back; drain every
            # one already readable and serve them as a single batch, so a
            # batching engine runs one forward pass per step, not one per seq.
            pending: list[tuple[bytes, SpeculateRequest]] = []
            while True:
                try:
                    identity, msg_type, corr_id, payload = self._sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if msg_type == _FREE:
                    self.engine.free(self._free_dec.decode(payload).seq_ids)
                    continue
                pending.append((identity, self._req_dec.decode(payload)))
            if not pending:
                continue
            t0 = time.perf_counter()
            results = speculate_many(self.engine, [r for _, r in pending])
            if _LOG:
                print(f"[draft {self.engine.node_id}] batch={len(pending)} "
                      f"{(time.perf_counter() - t0) * 1000.0:.1f}ms", flush=True)
            for (identity, request), result in zip(pending, results):
                if isinstance(result, Exception):
                    # must reach the client as a re-raise, not a silent timeout
                    self._sock.send_multipart(
                        [identity, _ERR, request.seq_id.encode(), str(result).encode()]
                    )
                else:
                    self._sock.send_multipart(
                        [identity, _SPEC, request.seq_id.encode(), self._enc.encode(result)]
                    )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sock.close(linger=0)
        self.engine.close()

    def __enter__(self) -> "DraftServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
