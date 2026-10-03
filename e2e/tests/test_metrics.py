"""Collector: accepted-count derivation and both metric conventions, on
hand-built records where every number can be checked by hand."""
from __future__ import annotations

import math

import pytest

from dsdreal.metrics.collector import attach_rounds, collect, summarize
from dsdreal.metrics.records import RequestRecord, RoundRecord

PROMPT = [1, 2, 3, 4]


def _round(seq, n_conf, t, n_draft=4, first=False):
    return RoundRecord(seq_id=seq, drafter_id="d0", gamma=4, n_draft=n_draft,
                       num_confirmed=n_conf, t_send_ms=t, t_recv_ms=t + 2, draft_ms=1.0,
                       injected_ms=0.5,
                       prompt_token_ids=PROMPT + [9] if first else None)


def _request(output_ts, arrival=0.0, rid="r0", prompt=PROMPT):
    return RequestRecord(request_id=rid, arrival_ms=arrival, prompt_token_ids=list(prompt),
                         max_tokens=len(output_ts), token_ts_ms=list(output_ts),
                         finish_ms=output_ts[-1])


def test_accepted_derivation_and_sim_metrics():
    # prefill at t=10 emits 1 token (confirmed=5). Rounds:
    #   r0 @10: next confirmed 8  -> delta 3, accepted 2, ends @20
    #   r1 @20: next confirmed 9  -> delta 1, accepted 0, ends @30
    #   r2 @30: final 4+10=14     -> delta 5, accepted 4, ends @finish=40
    rounds = [_round("s", 5, 10.0, first=True), _round("s", 8, 20.0), _round("s", 9, 30.0)]
    ts = [10.0] + [20.0] * 3 + [30.0] + [40.0] * 5
    (m,) = collect([_request(ts)], rounds)

    assert m.accounting_ok, m.notes
    assert [r.delta for r in m.rounds] == [3, 1, 5]
    assert [r.accepted for r in m.rounds] == [2, 0, 4]
    assert m.num_accepted == 6 and m.num_drafted == 12
    assert m.acceptance_ratio == pytest.approx(0.5)
    assert m.mean_acceptance_length == pytest.approx(3.0)       # with bonus

    assert m.ttft_ms == 10.0                                     # standard: first token
    assert m.tpot_ms == pytest.approx(30.0 / 9)
    assert m.ttft_sim_ms == 20.0                                 # end of first round with accepted>0
    # sim TPOT pools round time over accepted, skipping the 0-accepted round
    assert m.tpot_sim_ms == pytest.approx((10.0 + 10.0) / 6)


def test_phantom_last_round_is_dropped():
    """A round drafted on the step the request finished was never verified."""
    rounds = [_round("s", 5, 10.0, first=True), _round("s", 8, 20.0)]
    ts = [10.0] + [20.0] * 3
    (m,) = collect([_request(ts)], rounds)
    assert len(m.rounds) == 1 and m.accounting_ok


def test_bad_delta_flags_accounting():
    rounds = [_round("s", 5, 10.0, first=True), _round("s", 12, 20.0), _round("s", 13, 30.0)]
    ts = [10.0] + [20.0] * 7 + [30.0] * 2
    (m,) = collect([_request(ts)], rounds)
    assert not m.accounting_ok


def test_identical_prompts_match_in_admission_order():
    reqs = [_request([10.0, 20.0], arrival=0.0, rid="a"),
            _request([15.0, 25.0], arrival=1.0, rid="b")]
    rounds = [_round("s1", 5, 10.0, first=True), _round("s2", 5, 15.0, first=True)]
    by_req = attach_rounds(reqs, rounds)
    assert by_req["a"][0].seq_id == "s1" and by_req["b"][0].seq_id == "s2"


def test_summary_conventions():
    reqs = [_request([10.0 + i, 20.0 + i], arrival=float(i), rid=f"r{i}") for i in range(10)]
    s = summarize(collect(reqs, []), trim_frac=0.1)
    assert s["completed"] == 10
    assert s["sim"]["observed_span_ms"] == pytest.approx(29.0 - 0.0)
    assert s["sim"]["throughput_rps"] == pytest.approx(10 / 29.0 * 1000)
    # steady window drops the first and last request
    assert s["standard"]["throughput_rps"] == pytest.approx(8 / (28.0 - 1.0) * 1000)
    assert math.isnan(s["sim"]["tpot_ms_pooled"])  # no spec rounds at all


def test_last_verify_past_max_tokens_is_not_an_accounting_error():
    """The final verify accepted 4 + bonus with only 2 tokens of room: the
    runner's confirmed length (next round) counts all 5, the output keeps 2."""
    rounds = [_round("s", 5, 10.0, first=True), _round("s", 10, 20.0)]
    ts = [10.0, 20.0, 20.0]  # max_tokens = 3
    (m,) = collect([_request(ts)], rounds)
    assert m.accounting_ok, m.notes
    assert [r.accepted for r in m.rounds] == [4]
