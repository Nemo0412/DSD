"""HTTP/JSON draft server: the second transport, beside draft/server.py's ZMQ.

Same DraftEngine, same observable contract, different wire. It exists because
HTTP is trivially inspectable -- curl, any proxy, any language -- which is
worth a lot while debugging a distributed run, and because the payload is
tens of bytes of token ids, so JSON's cost is noise next to RTT and the
drafter's forward pass. Whether that stays true is not assumed: compare
`SpeculateReply.net_excess_ms` between the two transports and pick.

Wire, mirroring the ZMQ framing's three message kinds:

    POST /speculate  [SpeculateRequest, ...]  ->  200 [SpeculateReply, ...]
    POST /free       FreeSeq                  ->  204
    engine raised                             ->  400 {"error": "..."}

One POST carries a *list* of requests -- every request this step for this
drafter. ZMQ gets concurrency by pipelining many messages down one DEALER
socket; HTTP/1.1 has no usable pipelining, so a request-per-POST design would
serialise a batch's round trips on one connection. Batching per drafter gives
the same property structurally: one round trip per drafter per step, whatever
the batch size. See HttpDraftClient for the other half.

Two settings below are not cosmetic; without them the transport measures its
own misconfiguration instead of the network:
  - protocol_version = HTTP/1.1, or stdlib defaults to 1.0, closes the
    connection after every response, and every step pays a fresh TCP
    handshake -- an extra RTT that would land in net_excess_ms.
  - disable_nagle_algorithm, or small replies wait on the 40ms ACK timer.
    http.client already sets TCP_NODELAY on the client side; the server
    side does not.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec

from dsdreal.draft.base import DraftEngine, speculate_many
from dsdreal.proto.messages import FreeSeq, SpeculateRequest

_req_dec = msgspec.json.Decoder(list[SpeculateRequest])
_free_dec = msgspec.json.Decoder(FreeSeq)
_enc = msgspec.json.Encoder()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive; see module docstring
    disable_nagle_algorithm = True

    @property
    def engine(self) -> DraftEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _respond(self, code: int, payload: bytes | None) -> None:
        self.send_response(code)
        if payload is None:
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib's naming
        if self.path == "/free":
            self.engine.free(_free_dec.decode(self._body()).seq_ids)
            self._respond(204, None)
            return
        if self.path != "/speculate":
            self._respond(404, _enc.encode({"error": f"no route {self.path}"}))
            return

        requests = _req_dec.decode(self._body())
        self.server.speculate_posts += 1  # type: ignore[attr-defined]
        results = speculate_many(self.engine, requests)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            # Same reason the ZMQ server has an ERR frame: a drafter-side
            # desync has to surface as a client-side raise, not a hang.
            self._respond(400, _enc.encode({"error": str(errors[0])}))
            return
        self._respond(200, _enc.encode(results))

    def log_message(self, fmt: str, *args) -> None:
        pass  # stdlib logs every request to stderr; a per-step log is noise


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, engine: DraftEngine) -> None:
        self.engine = engine
        # Round trips actually served, not sequences. The batch-per-drafter
        # design is only worth anything if this stays at one per step, so it
        # is counted rather than assumed (tests assert on it).
        self.speculate_posts = 0
        super().__init__(addr, _Handler)


class HttpDraftServer:
    """One process per drafter node, same shape as DraftServer.

    `host`/`port` mirror ZMQ's `bind_addr`; pass port 0 and read
    `base_url` after start() for an ephemeral port.
    """

    def __init__(self, engine: DraftEngine, host: str = "127.0.0.1",
                 port: int = 0) -> None:
        self.engine = engine
        self._httpd = _Server((host, port), engine)
        self.host, self.port = self._httpd.server_address[:2]
        self.base_url = f"http://{self.host}:{self.port}"
        self._thread: threading.Thread | None = None

    @property
    def speculate_posts(self) -> int:
        """HTTP round trips served on /speculate, one per drafter per step."""
        return self._httpd.speculate_posts

    def start(self) -> "HttpDraftServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.05},
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd.server_close()
        self.engine.close()

    def __enter__(self) -> "HttpDraftServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
