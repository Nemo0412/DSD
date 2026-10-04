"""One target as a process the router can reach: an in-process LLMEngine
(same step loop and proposer wiring as experiments/driver.py) behind a ZMQ
ROUTER socket.

Messages (msgpack dicts, [kind, payload] frames after the identity):
  ADD    {request_id, arrival_ms, prompt_token_ids, max_tokens}  -> no reply
  STATS  {}                          -> {"queue_depth": waiting + running}
  DONE   {}                          -> after draining: {"requests": [...], "rounds": [...]}
  DEPTH  (target -> router, unsolicited) {"queue_depth": n, "adds": ADDs seen}
         after every step
         and every ADD, to every router seen so far. This is the "read-only
         snapshot" a routing policy reads: the router never blocks on a
         target to route (a STATS round trip waits for the target's current
         step -- tens of ms per request, which serialised JSQ admission).

The router stamps `arrival_ms` (absolute perf_counter ms) when the request
arrives at the *router*, so TTFT includes routing and the hop here. That is
only meaningful because router and targets share a machine: perf_counter is
CLOCK_MONOTONIC on Linux, one clock across processes.
"""
from __future__ import annotations

import dataclasses
import time

import msgspec
import zmq

from dsdreal.metrics.records import RequestRecord, RoundRecord

ADD, STATS, DONE, DEPTH = b"ADD", b"STATS", b"DONE", b"DEPTH"


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class TargetServer:
    def __init__(self, engine, make_params, bind: str, rounds: list[RoundRecord]) -> None:
        self.engine = engine
        self.make_params = make_params
        self.rounds = rounds
        self.records: dict[str, RequestRecord] = {}
        self._ctx = zmq.Context.instance()
        self.sock = self._ctx.socket(zmq.ROUTER)
        self.sock.bind(bind)
        self.bound_addr = self.sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder()
        self._peers: set[bytes] = set()
        self._adds = 0

    def queue_depth(self) -> int:
        return sum(1 for r in self.records.values() if r.finish_ms is None)

    def _publish_depth(self) -> None:
        payload = self._enc.encode({"queue_depth": self.queue_depth(), "adds": self._adds})
        for ident in self._peers:
            self.sock.send_multipart([ident, DEPTH, payload])

    def _handle(self, block_ms: int) -> bool:
        """Serve every message already readable (waiting up to block_ms for the
        first). Returns True once DONE was received."""
        done = False
        if not self.sock.poll(block_ms):
            return False
        while True:
            try:
                ident, kind, payload = self.sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return done
            msg = self._dec.decode(payload)
            self._peers.add(ident)
            if kind == ADD:
                self._adds += 1
                rec = RequestRecord(request_id=msg["request_id"], arrival_ms=msg["arrival_ms"],
                                    prompt_token_ids=list(msg["prompt_token_ids"]),
                                    max_tokens=msg["max_tokens"])
                self.records[rec.request_id] = rec
                self.engine.add_request(rec.request_id, {"prompt_token_ids": rec.prompt_token_ids},
                                        self.make_params(rec))
                self._publish_depth()
            elif kind == STATS:
                self.sock.send_multipart([ident, STATS, self._enc.encode(
                    {"queue_depth": self.queue_depth()})])
            elif kind == DONE:
                self._done_ident = ident
                done = True

    def serve(self) -> None:
        done = False
        while not done or self.engine.has_unfinished_requests():
            if self.engine.has_unfinished_requests():
                done = self._handle(0) or done
                outs = self.engine.step()
                ts = now_ms()
                for o in outs:
                    rec = self.records[o.request_id]
                    toks = o.outputs[0].token_ids
                    if len(toks) > rec.output_len:
                        rec.token_ts_ms.extend([ts] * (len(toks) - rec.output_len))
                        rec.output_token_ids = list(toks)
                    if o.finished and rec.finish_ms is None:
                        rec.finish_ms = ts
                self._publish_depth()
            else:
                done = self._handle(50) or done
        self.sock.send_multipart([self._done_ident, DONE, self._enc.encode({
            "requests": [dataclasses.asdict(r) for r in self.records.values()],
            "rounds": [dataclasses.asdict(r) for r in self.rounds],
        })])
        # let the reply flush before the socket goes away
        time.sleep(0.2)
        self.sock.close(linger=1000)


def main(argv: list[str] | None = None) -> int:
    """python -m dsdreal.router.target_server --bind tcp://127.0.0.1:5700 \\
           --ready-file runs/x/t0.ready [target args as in experiments/driver.py]"""
    import argparse

    from dsdreal.experiments.driver import add_target_args, build_target, setup_env, warmup

    ap = argparse.ArgumentParser()
    add_target_args(ap)
    ap.add_argument("--bind", required=True)
    ap.add_argument("--ready-file", default=None)
    args = ap.parse_args(argv)
    setup_env()
    target = build_target(args, ap.error)
    warmup(target, args.warmup)
    server = TargetServer(target.engine, target.make_params, args.bind, target.rounds)
    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write(server.bound_addr)
    print(f"target serving on {server.bound_addr}", flush=True)
    server.serve()
    target.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
