"""Router frontend + target servers over real ZMQ, no GPU: two TargetServers
in threads with a trivial engine (one token per request per step)."""
from __future__ import annotations

import threading
import time

import pytest

from dsdreal.metrics.collector import collect
from dsdreal.metrics.records import RequestRecord
from dsdreal.policy.routing import make_target_policy
from dsdreal.router.frontend import Router
from dsdreal.router.target_server import TargetServer
from dsdreal.workload.trace import WorkItem


class _Out:
    def __init__(self, rid, toks, finished):
        self.request_id, self.finished = rid, finished
        self.outputs = [type("C", (), {"token_ids": list(toks)})()]


class _Engine:
    """Each step emits one token for every live request; `step_s` per step."""

    def __init__(self, step_s=0.002):
        self.live: dict[str, list] = {}
        self.step_s = step_s

    def add_request(self, rid, prompt, max_tokens):
        self.live[rid] = [[], max_tokens]

    def has_unfinished_requests(self):
        return bool(self.live)

    def step(self):
        time.sleep(self.step_s)
        outs = []
        for rid, (toks, mx) in list(self.live.items()):
            toks.append(7)
            done = len(toks) >= mx
            outs.append(_Out(rid, toks, done))
            if done:
                del self.live[rid]
        return outs


def _start(step_s):
    srv = TargetServer(_Engine(step_s), lambda rec: rec.max_tokens, "tcp://127.0.0.1:0", [])
    th = threading.Thread(target=srv.serve, daemon=True)
    th.start()
    return srv, th


@pytest.mark.parametrize("policy", ["rr", "jsq"])
def test_router_dispatches_and_drains(policy):
    (a, ta), (b, tb) = _start(0.002), _start(0.002)
    router = Router({"t0": a.bound_addr, "t1": b.bound_addr}, make_target_policy(policy),
                    timeout_s=10)
    items = [WorkItem(f"r{i}", 3.0 * i, [1, 2, 3, i], 5) for i in range(20)]
    t0, decisions = router.run(items)
    results = router.drain()
    ta.join(5), tb.join(5)

    assert set(results) == {"t0", "t1"}
    reqs = [RequestRecord(**r) for res in results.values() for r in res["requests"]]
    assert sorted(r.request_id for r in reqs) == sorted(w.request_id for w in items)
    assert all(r.output_len == 5 and r.finish_ms is not None for r in reqs)
    # arrival is the router's scheduled time, so TTFT covers the hop
    by_id = {w.request_id: w for w in items}
    assert all(abs(r.arrival_ms - (t0 + by_id[r.request_id].arrival_ms)) < 1e-6 for r in reqs)
    per = collect(reqs, [])
    assert all(m.ttft_ms > 0 for m in per)
    counts = {t: sum(d["target_id"] == t for d in decisions.values()) for t in ("t0", "t1")}
    assert min(counts.values()) >= 5, counts
    if policy == "jsq":
        assert all(d["queue_depth_at_decision"] is not None for d in decisions.values())


def test_jsq_avoids_a_slow_target():
    (fast, tf), (slow, ts) = _start(0.001), _start(0.03)
    router = Router({"fast": fast.bound_addr, "slow": slow.bound_addr},
                    make_target_policy("jsq"), timeout_s=10)
    items = [WorkItem(f"r{i}", 2.0 * i, [1, 2, 3], 10) for i in range(30)]
    _, decisions = router.run(items)
    router.drain()
    tf.join(5), ts.join(5)
    to_fast = sum(d["target_id"] == "fast" for d in decisions.values())
    assert to_fast > 18, to_fast
