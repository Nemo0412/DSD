#!/usr/bin/env python3
"""Minimal unit checks for tree_accept parsing and replay."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from trace.tree_accept import TreeAcceptCursor, TreeRound
from trace.types import TraceRecord


def test_cursor_order_and_width() -> None:
    cur = TreeAcceptCursor.from_optional([
        {"round": 1, "depth": 5, "candidates": 10, "verified": 50, "tree_nodes": 60, "accepted_path_len": 0},
        {"round": 0, "depth": 5, "candidates": 16, "verified": 16, "accepted_path_len": 3},
    ])
    assert cur is not None and cur.remaining() == 2
    assert cur.next_round() == TreeRound(accepted=3, depth=5, tree_nodes=16)
    assert cur.next_round() == TreeRound(accepted=0, depth=5, tree_nodes=60)
    assert not cur.has_data()


def test_empty_is_none() -> None:
    assert TreeAcceptCursor.from_optional(None) is None
    assert TreeAcceptCursor.from_optional([]) is None


def test_trace_roundtrip() -> None:
    rounds = [{"round": 0, "depth": 4, "tree_nodes": 16, "accepted_path_len": 2}]
    rec = TraceRecord.from_dict({
        "arrival_ms": 0.0,
        "prompt_tokens": 16,
        "target_tokens": 3,
        "tree_accept": rounds,
    })
    cur = TreeAcceptCursor.from_optional(TraceRecord.from_dict(rec.to_dict()).tree_accept)
    assert cur is not None and cur.next_round() == TreeRound(accepted=2, depth=4, tree_nodes=16)


if __name__ == "__main__":
    test_cursor_order_and_width()
    test_empty_is_none()
    test_trace_roundtrip()
    print("tree_accept checks passed")
