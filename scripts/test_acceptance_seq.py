#!/usr/bin/env python3
"""Minimal unit checks for acceptance_seq parsing and replay."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from trace.acceptance_seq import AcceptanceSeqCursor, normalize_acceptance_seq
from trace.types import TraceRecord


def test_normalize() -> None:
    assert normalize_acceptance_seq([1, 0, 1]) == (1, 0, 1)
    assert normalize_acceptance_seq(None) == tuple()


def test_cursor_sd_semantics() -> None:
    cur = AcceptanceSeqCursor(sequence=(1, 1, 0, 1, 0, 1, 1, 1, 0))
    assert cur.verify(4) == (2, 2)  # accept 2, reject at 3rd, discard 4th
    assert cur.verify(4) == (1, 3)  # next bit is 1 then 0
    assert cur.verify(4) == (3, 1)


def test_trace_roundtrip() -> None:
    rec = TraceRecord.from_dict(
        {
            "arrival_ms": 0.0,
            "prompt_tokens": 16,
            "target_tokens": 32,
            "acceptance_seq": [1, 1, 0, 1],
        }
    )
    assert rec.acceptance_seq == (1, 1, 0, 1)
    assert TraceRecord.from_dict(rec.to_dict()).acceptance_seq == (1, 1, 0, 1)


if __name__ == "__main__":
    test_normalize()
    test_cursor_sd_semantics()
    test_trace_roundtrip()
    print("acceptance_seq checks passed")
