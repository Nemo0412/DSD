"""Workload: dsdSim trace parsing and token-exact prompt construction."""
from __future__ import annotations

from pathlib import Path

import pytest

from dsdreal.workload.trace import (
    TraceEntry, build_requests, load_prompt_texts, load_sim_trace, poisson_arrivals, retime,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "dsdsim"


def test_vendored_trace_parses():
    entries = load_sim_trace(DATA / "gsm8k_trace_master.jsonl")
    assert len(entries) == 400
    assert entries[0].arrival_ms == 0.0
    assert all(a.arrival_ms <= b.arrival_ms for a, b in zip(entries, entries[1:]))
    assert all(e.prompt_tokens > 0 and e.target_tokens > 0 for e in entries)
    assert len(load_prompt_texts(DATA / "gsm8k_trace_source.jsonl")) == 2000


def test_prompts_are_token_exact_and_distinct():
    entries = [TraceEntry(f"r{i}", float(i), 7, 5) for i in range(4)]
    texts = ["a b", "c d e", "f", "g h"]
    tokenize = lambda s: [ord(c) for c in s.split()[0]] * len(s.split())  # noqa: E731
    items = build_requests(entries, texts, tokenize, bos_token_id=1)
    for it in items:
        assert len(it.prompt_token_ids) == 7
        assert it.prompt_token_ids[0] == 1
        assert it.max_tokens == 5
    assert len({tuple(it.prompt_token_ids) for it in items}) == len(items)


def test_output_cap():
    entries = [TraceEntry("r0", 0.0, 3, 500)]
    (it,) = build_requests(entries, ["x"], lambda s: [9], max_output_tokens=64)
    assert it.max_tokens == 64 and it.prompt_token_ids == [9, 9, 9]


def test_poisson_rate_is_roughly_right():
    ts = poisson_arrivals(4000, rate_rps=20.0, seed=1)
    assert ts[0] == 0.0
    rate = (len(ts) - 1) / (ts[-1] / 1000.0)
    assert rate == pytest.approx(20.0, rel=0.08)


def test_retime_keeps_lengths():
    entries = [TraceEntry(f"r{i}", 25.0 * i, 10 + i, 20 + i) for i in range(10)]
    out = retime(entries, 5.0)
    assert [(e.prompt_tokens, e.target_tokens) for e in out] == \
        [(e.prompt_tokens, e.target_tokens) for e in entries]
    assert out[0].arrival_ms == 0.0 and out[-1].arrival_ms != entries[-1].arrival_ms


def test_wrap_keeps_exact_length():
    entries = [TraceEntry("r0", 0.0, 12, 5)]
    (it,) = build_requests(entries, ["x y z"], lambda s: [5] * 4, wrap=([1, 2, 3], [8, 9]),
                           min_body=1)
    assert len(it.prompt_token_ids) == 12
    assert it.prompt_token_ids[:3] == [1, 2, 3] and it.prompt_token_ids[-2:] == [8, 9]


def test_prompts_distinct_even_when_texts_collide():
    """Texts sharing a first token + a wrap leaving room for one body token
    used to yield identical prompts (seen on GPU: 22 of 400)."""
    entries = [TraceEntry(f"r{i}", float(i), 6, 5) for i in range(6)]
    texts = ["a b", "a c", "a d", "b e", "c f", "d g"]
    tok = lambda s: [ord(w) for w in s.split()]  # noqa: E731
    items = build_requests(entries, texts, tok, wrap=([1, 2], [9, 9]), min_body=2)
    assert len({tuple(i.prompt_token_ids) for i in items}) == 6
    assert all(len(i.prompt_token_ids) == 6 for i in items)


def test_short_trace_prompt_keeps_min_body_under_wrap():
    (it,) = build_requests([TraceEntry("r0", 0.0, 4, 5)], ["x"], lambda s: [5] * 30,
                           wrap=([1, 2, 3], [8, 9]))
    assert len(it.prompt_token_ids) == 3 + 16 + 2
