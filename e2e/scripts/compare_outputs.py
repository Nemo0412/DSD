#!/usr/bin/env python
"""Lossless gate for online-driver runs: compare outputs.jsonl of two runs
over the same trace (typically speculative vs no-draft baseline, fp32).

    python scripts/compare_outputs.py runs/base/outputs.jsonl runs/spec/outputs.jsonl

Exit 0 only if every request's tokens match exactly.
"""
from __future__ import annotations

import json
import sys


def load(path):
    return {d["request_id"]: d["token_ids"]
            for d in (json.loads(line) for line in open(path) if line.strip())}


def main(a_path: str, b_path: str) -> int:
    a, b = load(a_path), load(b_path)
    if set(a) != set(b):
        print(f"request sets differ: only-a={sorted(set(a) - set(b))[:5]} "
              f"only-b={sorted(set(b) - set(a))[:5]}")
        return 1
    bad = []
    for rid in sorted(a):
        x, y = a[rid], b[rid]
        if x != y:
            div = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            bad.append((rid, div, len(x), len(y)))
    print(json.dumps({"requests": len(a), "mismatched": len(bad),
                      "first_mismatches": bad[:10]}, indent=2))
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
