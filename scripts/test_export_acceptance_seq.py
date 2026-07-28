#!/usr/bin/env python3
"""Unit checks for hardware details → acceptance_seq export."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from acceptance.export_acceptance_seq import (  # noqa: E402
    details_record_to_trace,
    flatten_acceptance_seq,
    merge_acceptance_into_workload,
)
from trace.acceptance_seq import AcceptanceSeqCursor  # noqa: E402


def test_flatten_and_replay() -> None:
    iterations = [
        {"accepted_flags": [True, True, False]},
        {"accepted_flags": [True, False]},
        {"accepted_flags": [True, True, True, True]},
    ]
    seq = flatten_acceptance_seq(iterations)
    assert seq == (1, 1, 0, 1, 0, 1, 1, 1, 1)
    cursor = AcceptanceSeqCursor(sequence=seq)
    assert cursor.verify(4) == (2, 2)
    assert cursor.verify(4) == (1, 3)
    assert cursor.verify(4) == (4, 0)


def test_details_to_trace() -> None:
    record = {
        "prompt_index": 3,
        "prompt": "What is 2+2?",
        "total_generated_tokens": 7,
        "metadata": {"drafter_model": "draft", "verifier_model": "target"},
        "iterations": [
            {
                "context_length_before": 12,
                "accepted_flags": [True, True, False],
            },
            {
                "context_length_before": 15,
                "accepted_flags": [True, True, True, True],
            },
        ],
    }
    row = details_record_to_trace(record, arrival_ms=50.0)
    assert row["request_id"] == "profile_00003"
    assert row["prompt_tokens"] == 12
    assert row["target_tokens"] == 7
    assert row["acceptance_seq"] == [1, 1, 0, 1, 1, 1, 1]
    assert row["metadata"]["acceptance_source"] == "hardware_profile"


def test_merge_workload() -> None:
    workload = {
        "request_id": "gsm8k_00000",
        "arrival_ms": 0.0,
        "prompt_tokens": 36,
        "target_tokens": 114,
        "device_tier": "default",
        "metadata": {"client_id": "client_000"},
    }
    details = {
        "prompt_index": 0,
        "iterations": [{"accepted_flags": [True, False, True]}],
        "metadata": {"drafter_model": "d"},
    }
    merged = merge_acceptance_into_workload(workload, details)
    assert merged["request_id"] == "gsm8k_00000"
    assert merged["acceptance_seq"] == [1, 0, 1]
    assert merged["metadata"]["acceptance_source"] == "hardware_profile"
    assert merged["metadata"]["client_id"] == "client_000"


def test_cli_roundtrip() -> None:
    details = {
        "prompt_index": 0,
        "total_generated_tokens": 3,
        "metadata": {},
        "iterations": [
            {
                "context_length_before": 8,
                "accepted_flags": [True, True, False],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        details_path = tmp_path / "details.jsonl"
        out_path = tmp_path / "out.jsonl"
        details_path.write_text(json.dumps(details) + "\n", encoding="utf-8")

        # Invoke module logic directly (same as CLI core path).
        row = details_record_to_trace(details)
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        loaded = json.loads(out_path.read_text(encoding="utf-8").strip())
        assert loaded["acceptance_seq"] == [1, 1, 0]


if __name__ == "__main__":
    test_flatten_and_replay()
    test_details_to_trace()
    test_merge_workload()
    test_cli_roundtrip()
    print("export_acceptance_seq checks passed")
