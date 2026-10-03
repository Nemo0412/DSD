"""Router frontend for N:M: admits requests at their arrival times and sends
each to a target chosen by a routing policy (RR / Random / JSQ).

Targets are `dsdreal.router.target_server` processes. Arrival is stamped here
(scheduled time, absolute perf_counter ms, same machine => same clock), so
TTFT covers routing plus the hop to the target -- the "TTFT from router
entry" definition plan.md §5 fixes. JSQ reads the latest queue depth each
target pushed (after every step) plus the requests routed there since, never a
blocking query: a blocking STATS round trip waits for the target's current
step and serialised admission (seen on GPU: 3x TTFT under JSQ vs RR).

    python -m dsdreal.router.frontend --target-addr t0=tcp://127.0.0.1:5700 \\
        --target-addr t1=tcp://127.0.0.1:5701 --policy jsq --n-drafters 4 \\
        --per-draft-rps 1 --n-requests 400 --chat-template --out runs/nm/jsq
"""
from __future__ import annotations

import argparse
import json
import time

import msgspec
import zmq

from dsdreal.experiments.driver import _parse_addrs, add_workload_args, build_workload
from dsdreal.metrics.records import RequestRecord, RoundRecord
from dsdreal.router.target_server import ADD, DEPTH, DONE, STATS, now_ms


class Router:
    def __init__(self, addresses: dict[str, str], policy, timeout_s: float = 600.0) -> None:
        self.ids = list(addresses)
        self.policy = policy
        self.timeout_ms = int(timeout_s * 1000)
        ctx = zmq.Context.instance()
        self.socks = {}
        for tid, addr in addresses.items():
            s = ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(addr)
            self.socks[tid] = s
        self.enc = msgspec.msgpack.Encoder()
        self.dec = msgspec.msgpack.Decoder()
        # Latest pushed DEPTH per target and how many ADDs it had seen then;
        # requests routed after that are in flight and counted on top, so the
        # policy sees (nearly) the exact queue -- as dsdSim's routers, which
        # read the simulator's live state, do.
        self.snapshot = {t: 0 for t in self.ids}
        self.seen_adds = {t: 0 for t in self.ids}
        self.routed = {t: 0 for t in self.ids}

    def _on(self, tid: str, kind: bytes, payload: bytes) -> bool:
        """Handle an unsolicited message; False if it is something else."""
        if kind == DEPTH:
            d = self.dec.decode(payload)
            self.snapshot[tid], self.seen_adds[tid] = d["queue_depth"], d["adds"]
            return True
        return False

    def drain_depths(self) -> None:
        for tid, s in self.socks.items():
            while True:
                try:
                    kind, payload = s.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if not self._on(tid, kind, payload):
                    raise RuntimeError(f"unexpected {kind!r} from {tid}")

    def _ask_all(self, kind: bytes) -> dict[str, dict]:
        for s in self.socks.values():
            s.send_multipart([kind, self.enc.encode({})])
        out = {}
        for tid, s in self.socks.items():
            while tid not in out:
                if not s.poll(self.timeout_ms):
                    raise TimeoutError(f"target {tid} did not answer {kind!r}")
                k, payload = s.recv_multipart()
                if self._on(tid, k, payload):
                    continue
                assert k == kind, (k, kind)
                out[tid] = self.dec.decode(payload)
        return out

    def depths(self) -> dict[str, int]:
        """Fresh depths by a blocking round trip (tests, diagnostics). Routing
        uses the pushed snapshot instead -- see `run`."""
        return {t: r["queue_depth"] for t, r in self._ask_all(STATS).items()}

    def run(self, items) -> tuple[float, dict[str, dict]]:
        """-> (t0_ms, request_id -> routing decision)."""
        items = sorted(items, key=lambda w: w.arrival_ms)
        decisions: dict[str, dict] = {}
        t0 = now_ms()
        for w in items:
            while (wait := t0 + w.arrival_ms - now_ms()) > 0:
                self.drain_depths()
                time.sleep(min(wait, 1.0) / 1000.0)
            self.drain_depths()
            depths = ({t: self.snapshot[t] + self.routed[t] - self.seen_adds[t] for t in self.ids}
                      if self.policy.needs_depths else None)
            tid = self.policy.select_target(self.ids, depths)
            self.routed[tid] += 1
            self.socks[tid].send_multipart([ADD, self.enc.encode({
                "request_id": w.request_id, "arrival_ms": t0 + w.arrival_ms,
                "prompt_token_ids": list(w.prompt_token_ids), "max_tokens": w.max_tokens})])
            decisions[w.request_id] = {"target_id": tid,
                                       "queue_depth_at_decision": depths[tid] if depths else None}
        return t0, decisions

    def drain(self) -> dict[str, dict]:
        return self._ask_all(DONE)


def main(argv: list[str] | None = None) -> int:
    from dsdreal.metrics.collector import collect
    from dsdreal.metrics.export import write_run
    from dsdreal.policy.routing import make_target_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-addr", action="append", required=True, help="target_id=tcp://...")
    ap.add_argument("--policy", choices=("rr", "random", "jsq"), default="rr")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="tokenizer for building prompts")
    ap.add_argument("--n-drafters", type=int, default=1, help="for per_draft_rps load scaling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    add_workload_args(ap)
    args = ap.parse_args(argv)

    addresses = _parse_addrs(args.target_addr)
    items, rate = build_workload(args, args.n_drafters)
    router = Router(addresses, make_target_policy(args.policy, args.seed))
    t0, decisions = router.run(items)
    results = router.drain()

    per_request = []
    for tid, res in results.items():
        reqs = [RequestRecord(**r) for r in res["requests"]]
        rounds = [RoundRecord(**r) for r in res["rounds"]]
        per_request.extend(collect(reqs, rounds))
    per_request.sort(key=lambda m: m.arrival_ms)
    config = {k: v for k, v in vars(args).items()}
    config.update(rate_rps=rate, targets=addresses)
    system = write_run(args.out, per_request, t0, config, extra=decisions)
    counts = {t: sum(1 for d in decisions.values() if d["target_id"] == t) for t in addresses}
    print(json.dumps({"routed": counts, **system}, indent=2, default=str))
    return 0 if system.get("accounting_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
