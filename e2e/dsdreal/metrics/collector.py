"""Turn raw records into per-request and system metrics, in two conventions.

`standard` is what serving benchmarks report (vLLM's own definitions);
`sim` reproduces dsdSim's `Metrics.summary` (src/sim.py) so numbers can be
put side by side with the simulator's without a silent definitional gap:

| metric     | standard                                  | sim (dsdSim)                                          |
|------------|-------------------------------------------|-------------------------------------------------------|
| TTFT       | first output token - arrival              | end of the first round with >=1 accepted draft token  |
| TPOT       | (last - first token) / (n_out - 1), mean  | sum(round time) / sum(accepted), pooled over tokens;  |
|            | over requests                             | rounds with 0 accepted contribute nothing             |
| accepted   | mean_acceptance_length includes bonus     | accepted excludes the bonus/correction token          |
| throughput | completed / steady window (trim head/tail)| completed / (max finish - min arrival), no burn-in    |

A round's accepted count is not observable when it is proposed: it is the
growth of `num_confirmed` at the sequence's next round, minus the one
bonus/correction token the target always appends. A round's end time is
likewise the next round's start (propose() runs right after sampling the
verify step), or the request's finish for the last round.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

from dsdreal.metrics.records import RequestRecord, RoundRecord


@dataclass(slots=True)
class VerifiedRound:
    rec: RoundRecord
    delta: int          # tokens confirmed by this round's verify step (accepted + 1)
    accepted: int       # delta - 1, excludes the bonus token
    t_end_ms: float

    @property
    def dur_ms(self) -> float:
        return self.t_end_ms - self.rec.t_send_ms


@dataclass(slots=True)
class RequestMetrics:
    request_id: str
    drafter_id: str | None
    arrival_ms: float
    finish_ms: float
    prompt_len: int
    output_len: int
    rounds: list[VerifiedRound] = field(default_factory=list)
    accounting_ok: bool = True
    notes: list[str] = field(default_factory=list)

    # -- standard -----------------------------------------------------------
    ttft_ms: float = math.nan
    tpot_ms: float = math.nan
    e2e_ms: float = math.nan
    # -- sim ----------------------------------------------------------------
    ttft_sim_ms: float = math.nan
    tpot_sim_num_ms: float = 0.0   # sum of round time over rounds with accepted > 0
    tpot_sim_den: int = 0          # sum of accepted

    @property
    def tpot_sim_ms(self) -> float:
        return self.tpot_sim_num_ms / self.tpot_sim_den if self.tpot_sim_den else math.nan

    @property
    def num_accepted(self) -> int:
        return sum(r.accepted for r in self.rounds)

    @property
    def num_drafted(self) -> int:
        return sum(r.rec.n_draft for r in self.rounds)

    @property
    def acceptance_ratio(self) -> float:
        return self.num_accepted / self.num_drafted if self.num_drafted else math.nan

    @property
    def mean_acceptance_length(self) -> float:
        """vLLM's definition: includes the bonus token."""
        return sum(r.delta for r in self.rounds) / len(self.rounds) if self.rounds else math.nan


def attach_rounds(requests: list[RequestRecord],
                  rounds: list[RoundRecord]) -> dict[str, list[RoundRecord]]:
    """request_id -> its rounds, in call order.

    seq_id is the proposer's name for a sequence; the link back to the
    driver's request_id is the history sent on a sequence's first round,
    whose prefix is the request's prompt. Identical prompts are matched in
    admission order. A second "first round" for an already-matched prompt
    (preemption with recompute re-enters a slot as fresh) is attached to the
    most recent request with that prompt.
    """
    by_prompt: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for req in sorted(requests, key=lambda r: r.arrival_ms):
        by_prompt[tuple(req.prompt_token_ids)].append(req.request_id)
    lengths = sorted({len(k) for k in by_prompt}, reverse=True)
    unmatched = {k: deque(v) for k, v in by_prompt.items()}
    last_matched: dict[tuple[int, ...], str] = {}

    seq_to_req: dict[str, str] = {}
    out: dict[str, list[RoundRecord]] = defaultdict(list)
    for rec in rounds:
        if rec.seq_id not in seq_to_req:
            if rec.prompt_token_ids is None:
                continue  # never saw this sequence's first round
            hist = rec.prompt_token_ids
            key = next((tuple(hist[:n]) for n in lengths
                        if n <= len(hist) and tuple(hist[:n]) in unmatched), None)
            if key is None:
                continue
            if unmatched[key]:
                rid = unmatched[key].popleft()
                last_matched[key] = rid
            elif key in last_matched:
                rid = last_matched[key]
            else:
                continue
            seq_to_req[rec.seq_id] = rid
        out[seq_to_req[rec.seq_id]].append(rec)
    return dict(out)


def request_metrics(req: RequestRecord, recs: list[RoundRecord]) -> RequestMetrics:
    finish = req.finish_ms if req.finish_ms is not None else (
        req.token_ts_ms[-1] if req.token_ts_ms else math.nan)
    m = RequestMetrics(
        request_id=req.request_id,
        drafter_id=recs[0].drafter_id if recs else None,
        arrival_ms=req.arrival_ms, finish_ms=finish,
        prompt_len=req.prompt_len, output_len=req.output_len,
    )
    if req.token_ts_ms:
        m.ttft_ms = req.token_ts_ms[0] - req.arrival_ms
        m.e2e_ms = finish - req.arrival_ms
        if req.output_len > 1:
            m.tpot_ms = (req.token_ts_ms[-1] - req.token_ts_ms[0]) / (req.output_len - 1)

    final_len = req.prompt_len + req.output_len
    for i, rec in enumerate(recs):
        if i + 1 < len(recs):
            nxt = recs[i + 1]
            delta = nxt.num_confirmed - rec.num_confirmed
            t_end = nxt.t_send_ms
        else:
            delta = final_len - rec.num_confirmed
            t_end = finish - req.downlink_ms   # rounds are target-side times
            if delta <= 0:
                # Drafted for on the step the request finished: never
                # verified. Negative when that final verify accepted past
                # max_tokens -- the runner's confirmed length counts those
                # tokens, the output is then truncated to max_tokens. (The
                # previous round's delta keeps them: that is what the target
                # actually accepted.)
                break
            if delta > rec.n_draft + 1:
                m.accounting_ok = False
                m.notes.append(f"last round delta={delta} n_draft={rec.n_draft}")
        if i + 1 < len(recs) and not (1 <= delta <= rec.n_draft + 1):
            m.accounting_ok = False
            m.notes.append(f"round {i} delta={delta} n_draft={rec.n_draft}")
        accepted = max(0, min(delta - 1, rec.n_draft))
        m.rounds.append(VerifiedRound(rec=rec, delta=delta, accepted=accepted, t_end_ms=t_end))

    if recs and recs[0].num_confirmed != req.prompt_len + 1:
        m.accounting_ok = False
        m.notes.append(f"first round num_confirmed={recs[0].num_confirmed} "
                       f"!= prompt_len+1={req.prompt_len + 1}")

    for r in m.rounds:
        if r.accepted > 0:
            if math.isnan(m.ttft_sim_ms):
                m.ttft_sim_ms = r.t_end_ms + req.downlink_ms - req.arrival_ms
            m.tpot_sim_num_ms += r.dur_ms
            m.tpot_sim_den += r.accepted
    return m


def collect(requests: list[RequestRecord], rounds: list[RoundRecord]) -> list[RequestMetrics]:
    by_req = attach_rounds(requests, rounds)
    return [request_metrics(r, by_req.get(r.request_id, []))
            for r in sorted(requests, key=lambda r: r.arrival_ms)]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def _dist(xs: list[float]) -> dict[str, float]:
    return {"mean": _mean(xs), "p50": _pct(xs, 0.5), "p99": _pct(xs, 0.99)}


def summarize(per_request: list[RequestMetrics], trim_frac: float = 0.1) -> dict:
    done = [m for m in per_request if not math.isnan(m.finish_ms)]
    if not done:
        return {"completed": 0}
    span = max(m.finish_ms for m in done) - min(m.arrival_ms for m in done)

    ordered = sorted(done, key=lambda m: m.arrival_ms)
    k = int(len(ordered) * trim_frac)
    steady = ordered[k:len(ordered) - k] or ordered
    window = max(m.finish_ms for m in steady) - min(m.arrival_ms for m in steady)

    n_acc = sum(m.num_accepted for m in done)
    n_draft = sum(m.num_drafted for m in done)
    n_rounds = sum(len(m.rounds) for m in done)
    tpot_num = sum(m.tpot_sim_num_ms for m in done)
    tpot_den = sum(m.tpot_sim_den for m in done)
    out_tokens = sum(m.output_len for m in done)

    return {
        "completed": len(done),
        "accounting_ok": all(m.accounting_ok for m in done),
        "standard": {
            "ttft_ms": _dist([m.ttft_ms for m in done]),
            "tpot_ms": _dist([m.tpot_ms for m in done]),
            "e2e_ms": _dist([m.e2e_ms for m in done]),
            "throughput_rps": len(steady) / window * 1000.0 if window > 0 else math.nan,
            "throughput_tok_s": out_tokens / span * 1000.0 if span > 0 else math.nan,
            "steady_window_ms": window,
            "mean_acceptance_length": (sum(r.delta for m in done for r in m.rounds) / n_rounds
                                       if n_rounds else math.nan),
        },
        "sim": {
            "ttft_ms": _dist([m.ttft_sim_ms for m in done]),
            "tpot_ms_pooled": tpot_num / tpot_den if tpot_den else math.nan,
            "throughput_rps": len(done) / span * 1000.0 if span > 0 else math.nan,
            "observed_span_ms": span,
            "acceptance_ratio": n_acc / n_draft if n_draft else math.nan,
        },
    }
