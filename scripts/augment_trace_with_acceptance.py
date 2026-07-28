#!/usr/bin/env python3
"""Augment JSONL traces with acceptance_seq fields for replay-mode simulation.

Example:
  python scripts/augment_trace_with_acceptance.py \\
      --input traces/gsm8k_trace_1s.jsonl \\
      --output traces/gsm8k_trace_1s_with_acceptance.jsonl \\
      --gamma 4 --mean-accept 3.2 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def synthesize_acceptance_seq(
    *,
    target_tokens: int,
    gamma: int,
    mean_accept: float,
    rng: random.Random,
) -> List[int]:
    """Synthesize a draft-token acceptance/reject bitstring.

    This is a stand-in for hardware profiling. Prefer replacing sequences with
    real profiler output when available. The generator emits runs of accepted
    tokens terminated by a reject bit until roughly ``target_tokens`` accepts
    have been produced.
    """
    gamma = max(1, int(gamma))
    mean_accept = min(max(0.5, float(mean_accept)), float(gamma))
    p_continue = max(0.05, min(0.95, (mean_accept - 1.0) / max(1.0, gamma - 1.0)))

    seq: List[int] = []
    accepted = 0
    safety = 0
    while accepted < target_tokens and safety < target_tokens * 8:
        safety += 1
        run = 0
        while run < gamma and accepted + run < target_tokens:
            # Continue accepting with probability p_continue after the first token.
            if run == 0 or rng.random() < p_continue:
                seq.append(1)
                run += 1
            else:
                break
        accepted += run
        if accepted >= target_tokens:
            break
        # Reject terminates the speculation window.
        seq.append(0)
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--mean-accept", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            row = json.loads(text)
            if "acceptance_seq" not in row:
                target_tokens = int(row.get("target_tokens") or row.get("output_length") or 64)
                row["acceptance_seq"] = synthesize_acceptance_seq(
                    target_tokens=target_tokens,
                    gamma=args.gamma,
                    mean_accept=args.mean_accept,
                    rng=rng,
                )
                metadata = dict(row.get("metadata") or {})
                metadata["acceptance_source"] = "synthetic"
                row["metadata"] = metadata
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote acceptance-augmented trace to {args.output}")


if __name__ == "__main__":
    main()
