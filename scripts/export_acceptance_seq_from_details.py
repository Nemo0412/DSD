#!/usr/bin/env python3
"""Export hardware-profiled acceptance_seq traces from speculative_profiler details.

Examples:
  # Build a standalone replay trace from profiler details:
  python scripts/export_acceptance_seq_from_details.py \\
      --details results/run/gsm8k_test_details.jsonl \\
      --output traces/gsm8k_hardware_acceptance.jsonl \\
      --arrival-stride-ms 25

  # Attach sequences onto an existing workload JSONL (row-aligned):
  python scripts/export_acceptance_seq_from_details.py \\
      --details results/run/gsm8k_test_details.jsonl \\
      --workload traces/gsm8k_trace_1s.jsonl \\
      --output traces/gsm8k_trace_1s_hardware_acceptance.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from acceptance.export_acceptance_seq import (  # noqa: E402
    details_record_to_trace,
    iter_details_records,
    merge_acceptance_into_workload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        type=Path,
        required=True,
        help="JSONL from speculative_profiler.py --details-jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output workload JSONL with acceptance_seq",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        help="Optional existing workload JSONL to merge onto (same row order)",
    )
    parser.add_argument(
        "--arrival-stride-ms",
        type=float,
        default=25.0,
        help="Synthetic arrival spacing when not merging into a workload",
    )
    parser.add_argument(
        "--device-tier",
        default="default",
        help="device_tier field for standalone export",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of detail records to export",
    )
    args = parser.parse_args()

    if not args.details.exists():
        raise SystemExit(f"details file not found: {args.details}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    workload_rows = None
    if args.workload is not None:
        if not args.workload.exists():
            raise SystemExit(f"workload file not found: {args.workload}")
        with args.workload.open("r", encoding="utf-8") as handle:
            workload_rows = [
                json.loads(line)
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]

    written = 0
    with args.details.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for idx, record in enumerate(iter_details_records(src)):
            if args.limit is not None and written >= args.limit:
                break

            if workload_rows is not None:
                if idx >= len(workload_rows):
                    break
                row = merge_acceptance_into_workload(workload_rows[idx], record)
            else:
                row = details_record_to_trace(
                    record,
                    arrival_ms=float(idx) * float(args.arrival_stride_ms),
                    device_tier=args.device_tier,
                )
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} acceptance_seq traces to {args.output}")


if __name__ == "__main__":
    main()
