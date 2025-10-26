#!/usr/bin/env python3
"""
Prepare prompt files for real speculative-decoding runs.

This utility loads a Hugging Face dataset that was previously exported with
``datasets.save_to_disk`` and emits JSONL files for the train/validation/test
splits expected by the acceptance profiling pipeline.

Each JSON line contains:
    {
        "prompt": "...",
        "reference": "...",        # optional, only when --answer-column provided
        "metadata": {
            "dataset": "gsm8k",
            "index": 42
        }
    }

Usage example (matches the workflow documented in docs/DESIGN.md):

    python src/experiments/speculative/prepare_prompts.py \
        --dataset-name gsm8k \
        --dataset-path src/thirdparty/benchmarks/gsm8k \
        --split train \
        --text-column question \
        --answer-column answer \
        --train-size 500 \
        --val-size 100 \
        --test-size 100 \
        --train-output prompts/gsm8k_train.jsonl \
        --val-output prompts/gsm8k_val.jsonl \
        --test-output prompts/gsm8k_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset, load_from_disk

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None
    _HAS_TQDM = False


def _non_negative(value: Optional[int]) -> int:
    if value is None:
        return 0
    return max(0, int(value))


def _write_jsonl(
    path: Path,
    rows: Iterable[dict],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = rows
    use_progress = _HAS_TQDM and total and total > 1
    if use_progress:
        iterator = tqdm(rows, total=total, desc=desc or "Exporting prompts", unit="prompt")
    with path.open("w", encoding="utf-8") as handle:
        for row in iterator:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    if use_progress and hasattr(iterator, "close"):
        iterator.close()


def _slice(dataset: Dataset, indices: List[int]) -> Dataset:
    """Return a new dataset consisting of the given row indices."""
    if not indices:
        return dataset.select([])
    return dataset.select(indices)


def _build_records(
    dataset: Dataset,
    *,
    text_column: str,
    answer_column: Optional[str],
    dataset_name: str,
    offset: int,
) -> Iterable[dict]:
    for local_idx, example in enumerate(dataset):
        prompt = str(example.get(text_column, "") or "").strip()
        record = {
            "prompt": prompt,
            "metadata": {
                "dataset": dataset_name,
                "index": offset + local_idx,
            },
        }
        if answer_column:
            record["reference"] = str(example.get(answer_column, "") or "").strip()
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prompt JSONL splits.")
    parser.add_argument("--dataset-name", default="dataset", help="Label stored in metadata.")
    parser.add_argument("--dataset-path", required=True, help="Path for load_from_disk/save_to_disk.")
    parser.add_argument("--hf-dataset", help="Optional Hugging Face dataset identifier (e.g., cnn_dailymail).")
    parser.add_argument("--hf-config", help="Optional Hugging Face dataset config name (e.g., 3.0.0).")
    parser.add_argument("--split", default="train", help="Dataset split to read (default: train).")
    parser.add_argument("--text-column", required=True, help="Column containing the prompt/user text.")
    parser.add_argument("--answer-column", help="Optional column that stores reference completions.")
    parser.add_argument("--train-size", type=int, default=0, help="Number of examples for the train split (0 = all remaining).")
    parser.add_argument("--val-size", type=int, default=0, help="Number of examples for validation.")
    parser.add_argument("--test-size", type=int, default=0, help="Number of examples for testing.")
    parser.add_argument("--train-output", required=True, help="Output JSONL path for the train split.")
    parser.add_argument("--val-output", help="Output JSONL path for the validation split.")
    parser.add_argument("--test-output", required=True, help="Output JSONL path for the test split.")
    parser.add_argument("--shuffle-seed", type=int, default=1337, help="RNG seed used to shuffle rows before splitting.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if dataset_path.exists():
        dataset_dict = load_from_disk(dataset_path)
    else:
        if not args.hf_dataset:
            raise FileNotFoundError(
                f"Directory {dataset_path} not found and --hf-dataset not provided."
            )
        print(f"Loading Hugging Face dataset {args.hf_dataset} ({args.hf_config or 'default'})")
        dataset_dict = load_dataset(args.hf_dataset, args.hf_config or None)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching dataset under {dataset_path}")
        dataset_dict.save_to_disk(dataset_path)

    dataset = dataset_dict[args.split]
    total_rows = len(dataset)

    train_size = _non_negative(args.train_size)
    val_size = _non_negative(args.val_size)
    test_size = _non_negative(args.test_size)

    indices = list(range(total_rows))
    random.Random(args.shuffle_seed).shuffle(indices)

    def take(count: int) -> List[int]:
        nonlocal indices
        if count <= 0:
            return []
        selected, indices = indices[:count], indices[count:]
        return selected

    train_indices = take(train_size)
    val_indices = take(val_size)
    test_indices = take(test_size)

    if args.val_output and not val_indices:
        raise ValueError("Validation output requested but no validation samples allocated. Increase --val-size.")
    if not test_indices:
        raise ValueError("Test split is empty; please increase --test-size.")
    if not train_indices:
        raise ValueError("Train split is empty; please increase --train-size.")

    outputs = [
        (Path(args.train_output), train_indices, "train"),
        (Path(args.val_output), val_indices, "val") if args.val_output else None,
        (Path(args.test_output), test_indices, "test"),
    ]

    for entry in filter(None, outputs):
        out_path, split_indices, split_name = entry
        subset = _slice(dataset, split_indices)
        records = _build_records(
            subset,
            text_column=args.text_column,
            answer_column=args.answer_column,
            dataset_name=f"{args.dataset_name}:{split_name}",
            offset=0,
        )
        count = len(subset)
        _write_jsonl(
            out_path,
            records,
            total=count,
            desc=f"{args.dataset_name}:{split_name}",
        )
        print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
