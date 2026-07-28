"""Perturb acceptance_seq traces for genuine sensitivity studies.

Supports three modes requested for the Technical Supplement:
  1) acceptance-rate bias
  2) burst-length shorten/extend
  3) cross-workload / cross-split transfer

Example:
  python scripts/perturb_acceptance_seq.py \\
      --input traces/gsm8k_trace_1s_with_acceptance.jsonl \\
      --mode rate-bias --delta -0.2 --output /tmp/rate_m20.jsonl \\
      --report
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trace.acceptance_seq import AcceptanceSeqCursor, normalize_acceptance_seq  # noqa: E402


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def window_stats(seq: Sequence[int], gamma: int) -> Tuple[float, float, float]:
    """Return (accept_bit_rate, mean_accepted_per_window, mean_burst_of_ones)."""
    bits = list(normalize_acceptance_seq(seq))
    if not bits:
        return 0.0, 0.0, 0.0
    accept_rate = sum(bits) / len(bits)
    cursor = AcceptanceSeqCursor(sequence=tuple(bits))
    accepted_windows: List[int] = []
    while cursor.has_data():
        accepted, _rejected = cursor.verify(gamma)
        accepted_windows.append(accepted)
    mean_accept = (
        sum(accepted_windows) / len(accepted_windows) if accepted_windows else 0.0
    )

    bursts: List[int] = []
    run = 0
    for bit in bits:
        if bit == 1:
            run += 1
        elif run:
            bursts.append(run)
            run = 0
    if run:
        bursts.append(run)
    mean_burst = sum(bursts) / len(bursts) if bursts else 0.0
    return accept_rate, mean_accept, mean_burst


def corpus_stats(rows: Sequence[dict], gamma: int) -> Dict[str, float]:
    rates, accepts, bursts = [], [], []
    for row in rows:
        seq = row.get("acceptance_seq") or []
        if not seq:
            continue
        r, a, b = window_stats(seq, gamma)
        rates.append(r)
        accepts.append(a)
        bursts.append(b)
    n = max(1, len(rates))
    return {
        "n": float(len(rates)),
        "accept_bit_rate": sum(rates) / n if rates else 0.0,
        "mean_accepted_per_window": sum(accepts) / n if accepts else 0.0,
        "mean_burst_len": sum(bursts) / n if bursts else 0.0,
    }


def perturb_rate_bias(seq: Sequence[int], delta: float, rng: random.Random) -> List[int]:
    """Bias accept-bit rate by approximately ``delta`` (e.g. -0.2 => -20%)."""
    bits = list(normalize_acceptance_seq(seq))
    if not bits:
        return bits
    ones = [i for i, b in enumerate(bits) if b == 1]
    zeros = [i for i, b in enumerate(bits) if b == 0]
    target_ones = max(0, min(len(bits), int(round(len(ones) * (1.0 + delta)))))
    out = bits[:]
    if target_ones < len(ones):
        for idx in rng.sample(ones, len(ones) - target_ones):
            out[idx] = 0
    elif target_ones > len(ones) and zeros:
        need = min(len(zeros), target_ones - len(ones))
        for idx in rng.sample(zeros, need):
            out[idx] = 1
    return out


def perturb_burst(seq: Sequence[int], factor: float, rng: random.Random) -> List[int]:
    """Shorten (factor<1) or extend (factor>1) contiguous accepted runs."""
    bits = list(normalize_acceptance_seq(seq))
    if not bits:
        return bits
    out: List[int] = []
    i = 0
    while i < len(bits):
        if bits[i] == 0:
            out.append(0)
            i += 1
            continue
        j = i
        while j < len(bits) and bits[j] == 1:
            j += 1
        run = j - i
        if factor < 1.0:
            new_run = max(1, int(round(run * factor))) if run else 0
            out.extend([1] * new_run)
            # keep a reject if original run was terminated by 0
            if j < len(bits) and bits[j] == 0:
                out.append(0)
                i = j + 1
            else:
                i = j
        else:
            extra = max(0, int(round(run * (factor - 1.0))))
            # randomly insert extras before the terminating reject
            out.extend([1] * (run + extra))
            if j < len(bits) and bits[j] == 0:
                out.append(0)
                i = j + 1
            else:
                i = j
    # Keep roughly similar length budget by truncation/padding rejects.
    if len(out) > len(bits):
        out = out[: len(bits)]
    while len(out) < len(bits):
        out.append(0 if rng.random() < 0.5 else 1)
    return out


def apply_mode(
    rows: Sequence[dict],
    *,
    mode: str,
    delta: float,
    burst_factor: float,
    seed: int,
    donor_rows: Optional[Sequence[dict]] = None,
) -> List[dict]:
    rng = random.Random(seed)
    out_rows: List[dict] = []
    for idx, row in enumerate(rows):
        new_row = dict(row)
        seq = list(row.get("acceptance_seq") or [])
        metadata = dict(row.get("metadata") or {})
        if mode == "rate-bias":
            new_seq = perturb_rate_bias(seq, delta, rng)
            metadata["perturbation"] = {"mode": mode, "delta": delta, "seed": seed}
        elif mode == "burst":
            new_seq = perturb_burst(seq, burst_factor, rng)
            metadata["perturbation"] = {
                "mode": mode,
                "burst_factor": burst_factor,
                "seed": seed,
            }
        elif mode == "cross-split":
            if not donor_rows:
                raise ValueError("cross-split requires donor rows")
            donor = donor_rows[idx % len(donor_rows)]
            new_seq = list(donor.get("acceptance_seq") or [])
            metadata["perturbation"] = {
                "mode": mode,
                "donor_request_id": donor.get("request_id"),
                "seed": seed,
            }
        else:
            raise ValueError(f"unknown mode: {mode}")
        new_row["acceptance_seq"] = new_seq
        new_row["metadata"] = metadata
        out_rows.append(new_row)
    return out_rows


def first_order_system_delta(
    base_tpot: float,
    base_thr: float,
    base_accept: float,
    new_accept: float,
) -> Tuple[float, float]:
    """Map mean-accepted-per-window changes to first-order TPOT/throughput.

    More accepted tokens / window => fewer speculation rounds => lower TPOT and
    higher throughput. This is intentionally a request-level approximation used
    only after bit-level perturbations have been applied.
    """
    if base_accept <= 1e-9:
        return base_tpot, base_thr
    scale = base_accept / max(1e-9, new_accept)
    tpot = base_tpot * scale
    thr = base_thr / scale
    return tpot, thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["rate-bias", "burst", "cross-split"],
        required=True,
    )
    parser.add_argument("--delta", type=float, default=0.0, help="For rate-bias")
    parser.add_argument(
        "--burst-factor",
        type=float,
        default=1.0,
        help="For burst: <1 shorten, >1 extend",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        help="Donor JSONL for cross-split transfer",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--base-tpot", type=float, default=19.5)
    parser.add_argument("--base-thr", type=float, default=155.1)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    donor_rows = load_jsonl(args.donor) if args.donor else None
    if args.mode == "cross-split" and donor_rows is None:
        # Default: first half donors, second half recipients (and vice versa via CLI).
        mid = max(1, len(rows) // 2)
        donor_rows = rows[:mid]
        rows = rows[mid:] or rows[:mid]

    base = corpus_stats(rows if args.mode != "cross-split" else load_jsonl(args.input), args.gamma)
    # For fair comparison, stats before perturbation on the evaluated rows.
    eval_base = corpus_stats(rows, args.gamma)
    out_rows = apply_mode(
        rows,
        mode=args.mode,
        delta=args.delta,
        burst_factor=args.burst_factor,
        seed=args.seed,
        donor_rows=donor_rows,
    )
    write_jsonl(args.output, out_rows)
    new = corpus_stats(out_rows, args.gamma)

    if args.report:
        tpot, thr = first_order_system_delta(
            args.base_tpot,
            args.base_thr,
            eval_base["mean_accepted_per_window"],
            new["mean_accepted_per_window"],
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "delta": args.delta,
                    "burst_factor": args.burst_factor,
                    "n_eval": new["n"],
                    "base": eval_base,
                    "perturbed": new,
                    "first_order": {"tpot_ms": tpot, "throughput": thr},
                    "reference_corpus": base,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
