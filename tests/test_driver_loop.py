"""run_open_loop + RemoteDraftProposer + collector end to end, no GPU.

`_FakeSpecEngine` mimics the parts of vLLM the driver and proposer see: a
fixed number of batch slots reused as requests finish, a prefill step that
samples one token, then verify steps that confirm (accepted + 1) tokens and
call `propose()` with the same three arrays gpu_model_runner passes. That
exercises the positional slot tracking, the seq_id -> request_id join and
the accounting identity together, including slot reuse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.experiments.driver import run_open_loop
from dsdreal.metrics.collector import collect, summarize
from dsdreal.metrics.records import RoundRecord
from dsdreal.policy.routing import RoundRobin
from dsdreal.proto.transport import InProcessDraftClient
from dsdreal.target import proposer
from dsdreal.workload.trace import WorkItem

STEP_MS = 10.0


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s * 1000.0


@dataclass
class _Req:
    rid: str
    tokens: list[int]
    prompt_len: int
    max_tokens: int
    draft: list[int]
    prefilled: bool = False
    finished: bool = False

    @property
    def out(self):
        return self.tokens[self.prompt_len:]


class _Out:
    def __init__(self, r: _Req):
        self.request_id = r.rid
        self.finished = r.finished
        self.outputs = [type("C", (), {"token_ids": list(r.out)})()]


class _FakeSpecEngine:
    def __init__(self, clock, slots=2, gamma=3, max_len=512):
        self.clock, self.gamma, self.max_len = clock, gamma, max_len
        self.slots: list[_Req | None] = [None] * slots
        self.waiting: list[_Req] = []
        self.proposer = proposer.RemoteDraftProposer(
            type("V", (), {"speculative_config": type("S", (), {"num_speculative_tokens": gamma})()})())
        self.n_step = 0

    def add_request(self, rid, prompt, params):
        toks = list(prompt["prompt_token_ids"])
        self.waiting.append(_Req(rid, toks, len(toks), params, []))

    def has_unfinished_requests(self):
        return bool(self.waiting) or any(s is not None for s in self.slots)

    def step(self):
        self.n_step += 1
        for i, s in enumerate(self.slots):
            if s is None and self.waiting:
                self.slots[i] = self.waiting.pop(0)
        self.clock.t += STEP_MS
        sampled = [[] for _ in self.slots]
        outs = []
        for i, r in enumerate(self.slots):
            if r is None:
                continue
            if not r.prefilled:
                new = [7000 + len(r.tokens)]
                r.prefilled = True
            else:
                acc = (self.n_step + i) % (len(r.draft) + 1)
                new = r.draft[:acc] + [8000 + len(r.tokens)]
            room = r.max_tokens - len(r.out)
            new = new[:room]
            r.tokens.extend(new)
            sampled[i] = new
            r.finished = len(r.out) >= r.max_tokens
            outs.append(_Out(r))

        n = len(self.slots)
        num_tokens = np.zeros(n, dtype=np.int64)
        tok = np.zeros((n, self.max_len), dtype=np.int64)
        for i, r in enumerate(self.slots):
            if r is not None:
                num_tokens[i] = len(r.tokens)
                tok[i, :len(r.tokens)] = r.tokens
        drafts = self.proposer.propose(sampled, num_tokens, tok)
        for i, r in enumerate(self.slots):
            if r is not None:
                r.draft = drafts[i]
                if r.finished:
                    self.slots[i] = None
        return outs


@pytest.fixture(autouse=True)
def _reset():
    yield
    proposer.configure(None)


def _items(n, prompt_len, out_len, gap_ms):
    return [WorkItem(f"r{i}", i * gap_ms, [100 + i] * prompt_len, out_len) for i in range(n)]


def test_open_loop_end_to_end_accounting():
    clock = _Clock()
    rounds: list[RoundRecord] = []
    client = InProcessDraftClient({"d0": EchoDraftEngine("d0"), "d1": EchoDraftEngine("d1")})
    proposer.configure(client, routing=RoundRobin(), rounds=rounds)
    engine = _FakeSpecEngine(clock, slots=2, gamma=3)

    items = _items(6, prompt_len=5, out_len=12, gap_ms=25.0)
    t0, reqs = run_open_loop(engine, items, lambda w: w.max_tokens,
                             clock=clock, sleep=clock.sleep)

    assert all(r.output_len == 12 and r.finish_ms is not None for r in reqs)
    per = collect(reqs, rounds)
    for m in per:
        assert m.accounting_ok, (m.request_id, m.notes)
        assert m.rounds, m.request_id
        # every output token after the prefill one came from a verified round
        assert sum(r.delta for r in m.rounds) == m.output_len - 1
        assert m.ttft_ms >= STEP_MS
    assert {m.drafter_id for m in per} == {"d0", "d1"}
    s = summarize(per)
    assert s["completed"] == 6 and s["accounting_ok"]
    assert 0.0 < s["sim"]["acceptance_ratio"] < 1.0


def test_idle_gap_sleeps_until_next_arrival():
    clock = _Clock()
    proposer.configure(InProcessDraftClient(EchoDraftEngine("d0")))
    engine = _FakeSpecEngine(clock, slots=1)
    items = _items(2, prompt_len=3, out_len=2, gap_ms=1000.0)
    t0, reqs = run_open_loop(engine, items, lambda w: w.max_tokens,
                             clock=clock, sleep=clock.sleep)
    second = next(r for r in reqs if r.request_id == "r1")
    assert second.arrival_ms == t0 + 1000.0
    assert second.token_ts_ms[0] - second.arrival_ms == pytest.approx(STEP_MS)


def test_slot_reuse_with_longer_new_prompt_is_a_new_request():
    """Regression: a new request in a reused slot whose prompt+1 is not
    shorter than the previous occupant's final length must still be
    detected as fresh, not continued (drafter desync otherwise)."""
    clock = _Clock()
    rounds: list[RoundRecord] = []
    proposer.configure(InProcessDraftClient(EchoDraftEngine("d0")), rounds=rounds)
    engine = _FakeSpecEngine(clock, slots=1, gamma=2)
    items = [WorkItem("short", 0.0, [5] * 3, 3), WorkItem("long", 1.0, [6] * 20, 4)]
    t0, reqs = run_open_loop(engine, items, lambda w: w.max_tokens,
                             clock=clock, sleep=clock.sleep)
    per = {m.request_id: m for m in collect(reqs, rounds)}
    assert per["long"].rounds and per["long"].accounting_ok, per["long"].notes


def test_client_side_links_shift_ttft_not_tpot(monkeypatch):
    """dsdSim's scenario: the prompt pays an uplink before admission, every
    token a downlink before the client sees it. TTFT (both conventions) grows
    by exactly up + down; TPOT and the accounting do not change."""
    def run(up, down):
        clock = _Clock()
        # round timestamps come from the transport's perf_counter: put it on
        # the same fake clock, as the real system shares one clock
        import dsdreal.proto.transport as transport
        monkeypatch.setattr(transport.time, "perf_counter", lambda: clock.t / 1000.0)
        rounds: list[RoundRecord] = []
        proposer.configure(InProcessDraftClient(EchoDraftEngine("d0")), rounds=rounds)
        engine = _FakeSpecEngine(clock, slots=1, gamma=3)
        items = _items(1, prompt_len=5, out_len=12, gap_ms=0.0)
        _, reqs = run_open_loop(engine, items, lambda w: w.max_tokens, clock=clock,
                                sleep=clock.sleep, uplink_ms=up, downlink_ms=down)
        (m,) = collect(reqs, rounds)
        proposer.configure(None)
        return m

    base, edge = run(0.0, 0.0), run(5.0, 5.0)
    assert edge.accounting_ok and base.accounting_ok
    assert edge.ttft_ms == pytest.approx(base.ttft_ms + 10.0)
    assert edge.ttft_sim_ms == pytest.approx(base.ttft_sim_ms + 10.0)
    assert edge.e2e_ms == pytest.approx(base.e2e_ms + 10.0)
    assert edge.tpot_ms == pytest.approx(base.tpot_ms)
    assert edge.tpot_sim_ms == pytest.approx(base.tpot_sim_ms)
