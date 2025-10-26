#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Train the latency-head dynamic gamma predictor (single residual MLP) on gamma-oracle data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict, deque

DEFAULT_DATASETS = [
    Path("data/gamma_oracle/raw/gsm8k"),
    Path("data/gamma_oracle/raw/cnndm"),
    Path("data/gamma_oracle/raw/humaneval"),
]

FEATURE_FIELDS = [
    "queue_util",
    "queue_trend",
    "pending_norm",
    "rtt_avg_norm",
    "rtt_delta",
    "tpot_avg_norm",
    "tpot_delta",
    "drafter_capability_norm",
    "acceptance_recent",
    "last_gamma_norm",
    "context_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP (1 residual block) gamma predictor from oracle datasets.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Directories containing raw gamma-oracle logs (default: gsm8k/cnndm/humaneval).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination JSON file storing the trained head weights/biases.",
    )
    parser.add_argument(
        "--torch-checkpoint",
        type=Path,
        default=None,
        help="Optional path to save the raw PyTorch checkpoint.",
    )
    parser.add_argument(
        "--default-gamma",
        type=float,
        default=4.0,
        help="Baseline gamma added to the MLP output delta (default 4.0).",
    )
    parser.add_argument(
        "--min-gamma",
        type=float,
        default=1.0,
        help="Lower clamp bound for gamma predictions (default 1).",
    )
    parser.add_argument(
        "--max-gamma",
        type=float,
        default=8.0,
        help="Upper clamp bound for gamma predictions (default 8).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden dimension for the residual block (default 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size (default 512).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Adam learning rate (default 3e-4).",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=3,
        help="Number of previous jobs to retain for history features (default 3).",
    )
    parser.add_argument(
        "--acceptance-alpha",
        type=float,
        default=0.4,
        help="EMA coefficient for recent acceptance (default 0.4).",
    )
    parser.add_argument(
        "--pending-ref",
        type=float,
        default=2048.0,
        help="Reference tokens for pending normalization (default 2048).",
    )
    parser.add_argument(
        "--capability-ref",
        type=float,
        default=1.0,
        help="Reference drafter capability (default 1).",
    )
    parser.add_argument(
        "--context-ref",
        type=float,
        default=2048.0,
        help="Reference context tokens (default 2048).",
    )
    parser.add_argument(
        "--rtt-ref-ms",
        type=float,
        default=30.0,
        help="Reference RTT milliseconds for normalization (default 30).",
    )
    parser.add_argument(
        "--tpot-ref-ms",
        type=float,
        default=15.0,
        help="Reference TPOT milliseconds for normalization (default 15).",
    )
    parser.add_argument(
        "--ttft-slo",
        type=float,
        default=None,
        help="Optional TTFT SLO threshold used when selecting oracle actions.",
    )
    parser.add_argument(
        "--tpot-slo",
        type=float,
        default=None,
        help="Optional TPOT SLO threshold (max of samples).",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc


def _discover_scenario_labels(raw_dir: Path) -> List[str]:
    labels: set[str] = set()
    for path in raw_dir.glob("scenario_*_gamma*.jsonl"):
        stem = path.stem
        base = stem.split("_gamma")[0]
        label = base.replace("scenario_", "")
        labels.add(label)
    return sorted(labels)


def _load_candidate_runs(raw_dir: Path, scenario_label: str) -> Tuple[Dict[str, Dict[int, dict]], List[str]]:
    pattern = f"scenario_{scenario_label}_gamma*.jsonl"
    candidate_map: Dict[str, Dict[int, dict]] = {}
    order: List[str] = []
    first_path: Optional[Path] = None

    for path in sorted(raw_dir.glob(pattern)):
        if first_path is None:
            first_path = path
        gamma_str = path.stem.split("gamma")[-1]
        try:
            gamma = int(gamma_str)
        except ValueError:
            continue
        for line in _load_jsonl(path):
            request_id = line.get("request_id")
            if request_id is None:
                continue
            candidate_map.setdefault(request_id, {})[gamma] = line

    if first_path is not None:
        with first_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                rid = payload.get("request_id")
                if rid is not None:
                    order.append(rid)

    return candidate_map, order


def _passes_slo(record: dict, ttft_slo: Optional[float], tpot_slo: Optional[float]) -> bool:
    metrics = record.get("metrics") or {}
    if ttft_slo is not None:
        ttft = metrics.get("ttft_ms")
        if ttft is None or float(ttft) > ttft_slo:
            return False
    if tpot_slo is not None:
        samples = metrics.get("tpot_samples") or []
        if not samples:
            return False
        if max(float(s) for s in samples) > tpot_slo:
            return False
    return True


def _pick_best(candidates: Mapping[int, dict], ttft_slo: Optional[float], tpot_slo: Optional[float]) -> Optional[Tuple[int, dict]]:
    best_gamma = None
    best_record = None
    best_duration = float("inf")
    for gamma, record in candidates.items():
        if not _passes_slo(record, ttft_slo, tpot_slo):
            continue
        metrics = record.get("metrics") or {}
        duration = float(metrics.get("duration_ms", float("inf")))
        if duration < best_duration:
            best_duration = duration
            best_gamma = gamma
            best_record = record
    if best_gamma is None or best_record is None:
        return None
    return best_gamma, best_record


def _mean(values: deque) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_field(history: deque, field: str) -> float:
    if not history:
        return 0.0
    vals = [entry.get(field, 0.0) for entry in history if entry.get(field, 0.0)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _delta_field(history: deque, field: str) -> float:
    if not history or len(history) < 2:
        return 0.0
    last = history[-1].get(field, 0.0)
    prev = history[-2].get(field, last)
    return last - prev


class ScenarioState:
    def __init__(
        self,
        *,
        history_window: int,
        acceptance_alpha: float,
        default_gamma: float,
        max_gamma: float,
    ) -> None:
        self.history_window = history_window
        self.acceptance_alpha = max(0.0, min(1.0, acceptance_alpha))
        self.default_gamma = default_gamma
        self.max_gamma = max_gamma
        self.target_queue_history: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.connection_history: DefaultDict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.recent_acceptance: Dict[str, float] = {}
        self.last_gamma: Dict[Tuple[str, str], float] = {}

    def build_feature_row(
        self,
        record: dict,
        *,
        pending_ref: float,
        capability_ref: float,
        context_ref: float,
        rtt_ref_ms: float,
        tpot_ref_ms: float,
    ) -> Optional[List[float]]:
        features = record.get("features")
        metrics = record.get("metrics")
        draft_id = record.get("draft_id")
        target_id = record.get("target_id")
        if not features or not metrics or not draft_id or not target_id:
            return None

        queue_util = float(features.get("queue_utilization", 0.0))
        queue_hist = self.target_queue_history.get(target_id)
        queue_avg = _mean(queue_hist) if queue_hist else queue_util
        queue_trend = queue_util - queue_avg if queue_hist else 0.0

        pending_tokens = float(features.get("pending_tokens", 0.0))
        workload_util = float(features.get("target_workload_util", 0.0))
        if workload_util <= 0.0:
            batch = float(features.get("target_batch_size", 0.0))
            denom = batch * max(1.0, self.default_gamma) if batch > 0 else max(1.0, self.default_gamma)
            workload_util = pending_tokens / denom if denom > 0 else 0.0
        workload_util = max(0.0, min(2.0, workload_util))

        conn_key = (draft_id, target_id)
        conn_hist = self.connection_history.get(conn_key)
        rtt_avg = _mean_field(conn_hist, "avg_rtt")
        rtt_delta = _delta_field(conn_hist, "avg_rtt")
        tpot_avg = _mean_field(conn_hist, "avg_tpot")
        tpot_delta = _delta_field(conn_hist, "avg_tpot")

        capability_norm = float(features.get("drafter_capability", 0.0)) / max(1e-6, capability_ref)
        acceptance_recent = self.recent_acceptance.get(target_id, float(features.get("acceptance_estimate", 0.0)))
        last_gamma = self.last_gamma.get(conn_key, self.default_gamma)

        row = [
            queue_util,
            queue_trend,
            workload_util,
            (rtt_avg / rtt_ref_ms) if rtt_ref_ms > 0 else rtt_avg,
            (rtt_delta / rtt_ref_ms) if rtt_ref_ms > 0 else rtt_delta,
            (tpot_avg / tpot_ref_ms) if tpot_ref_ms > 0 else tpot_avg,
            (tpot_delta / tpot_ref_ms) if tpot_ref_ms > 0 else tpot_delta,
            capability_norm,
            acceptance_recent,
            last_gamma / max(1.0, self.max_gamma),
            float(features.get("context_length", 0.0)) / max(1.0, context_ref),
        ]
        return row

    def update(self, record: dict, gamma: float) -> None:
        features = record.get("features") or {}
        metrics = record.get("metrics") or {}
        draft_id = record.get("draft_id")
        target_id = record.get("target_id")
        if not draft_id or not target_id:
            return

        queue_util = float(features.get("queue_utilization", 0.0))
        if queue_util > 0.0:
            self.target_queue_history[target_id].append(queue_util)

        conn_key = (draft_id, target_id)
        self.connection_history[conn_key].append(
            {
                "avg_rtt": float(metrics.get("avg_rtt_ms", 0.0)),
                "avg_tpot": float(metrics.get("avg_tpot_ms", 0.0)),
            }
        )

        acceptance = float(features.get("acceptance_estimate", 0.0))
        if acceptance > 0.0:
            prev = self.recent_acceptance.get(target_id)
            if prev is None:
                self.recent_acceptance[target_id] = acceptance
            else:
                alpha = self.acceptance_alpha
                self.recent_acceptance[target_id] = alpha * acceptance + (1.0 - alpha) * prev

        self.last_gamma[conn_key] = float(gamma)


def build_feature_matrix(
    raw_dirs: Sequence[Path],
    *,
    pending_ref: float,
    capability_ref: float,
    context_ref: float,
    rtt_ref_ms: float,
    tpot_ref_ms: float,
    default_gamma: float,
    max_gamma: float,
    history_window: int,
    acceptance_alpha: float,
    ttft_slo: Optional[float],
    tpot_slo: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:

    rows: List[List[float]] = []
    labels: List[float] = []

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        for scenario_label in _discover_scenario_labels(raw_dir):
            candidates, order = _load_candidate_runs(raw_dir, scenario_label)
            if not candidates:
                continue
            state = ScenarioState(
                history_window=history_window,
                acceptance_alpha=acceptance_alpha,
                default_gamma=default_gamma,
                max_gamma=max_gamma,
            )
            for request_id in order:
                per_request = candidates.get(request_id)
                if not per_request:
                    continue
                best = _pick_best(per_request, ttft_slo, tpot_slo)
                if best is None:
                    continue
                best_gamma, record = best
                row = state.build_feature_row(
                    record,
                    pending_ref=pending_ref,
                    capability_ref=capability_ref,
                    context_ref=context_ref,
                    rtt_ref_ms=rtt_ref_ms,
                    tpot_ref_ms=tpot_ref_ms,
                )
                if row is None:
                    continue
                rows.append(row)
                labels.append(float(best_gamma))
                state.update(record, best_gamma)

    if not rows:
        raise ValueError("No training samples found in the provided raw directories.")

    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


class ResidualGammaHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.block1 = nn.Linear(hidden_dim, hidden_dim)
        self.block2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.proj(x)
        block = self.block2(self.activation(self.block1(proj)))
        hidden = proj + block
        return self.out(hidden).squeeze(-1)


def train_model(
    model: ResidualGammaHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    min_gamma: float,
    max_gamma: float,
) -> Dict[str, float]:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            preds = torch.clamp(preds, min=min_gamma, max=max_gamma)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.clamp(model(features), min=min_gamma, max=max_gamma)
        mae = torch.mean(torch.abs(preds - labels)).item()
        rmse = torch.sqrt(torch.mean((preds - labels) ** 2)).item()
    return {"mae": mae, "rmse": rmse}


def _tensor_to_list(tensor: torch.Tensor) -> List:
    return tensor.detach().cpu().tolist()


def export_state(
    model: ResidualGammaHead,
    output_path: Path,
    *,
    metadata: Dict[str, float],
) -> None:
    state = {
        "feature_order": FEATURE_FIELDS,
        "proj": {
            "weight": _tensor_to_list(model.proj.weight),
            "bias": _tensor_to_list(model.proj.bias),
        },
        "block1": {
            "weight": _tensor_to_list(model.block1.weight),
            "bias": _tensor_to_list(model.block1.bias),
        },
        "block2": {
            "weight": _tensor_to_list(model.block2.weight),
            "bias": _tensor_to_list(model.block2.bias),
        },
        "output": {
            "weight": _tensor_to_list(model.out.weight.squeeze(0)),
            "bias": float(model.out.bias.item()),
        },
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def main() -> None:
    args = parse_args()
    features, labels = build_feature_matrix(
        args.datasets,
        pending_ref=args.pending_ref,
        capability_ref=args.capability_ref,
        context_ref=args.context_ref,
        rtt_ref_ms=args.rtt_ref_ms,
        tpot_ref_ms=args.tpot_ref_ms,
        default_gamma=args.default_gamma,
        max_gamma=args.max_gamma,
        history_window=args.history_window,
        acceptance_alpha=args.acceptance_alpha,
        ttft_slo=args.ttft_slo,
        tpot_slo=args.tpot_slo,
    )

    model = ResidualGammaHead(len(FEATURE_FIELDS), args.hidden_dim)
    metrics = train_model(
        model,
        features,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        min_gamma=args.min_gamma,
        max_gamma=args.max_gamma,
    )

    export_state(
        model,
        args.output_json,
        metadata={
            "datasets": [str(p) for p in args.datasets],
            "default_gamma": args.default_gamma,
            "min_gamma": args.min_gamma,
            "max_gamma": args.max_gamma,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )

    if args.torch_checkpoint:
        args.torch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, args.torch_checkpoint)

    print(
        f"Training complete. MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}. "
        f"Weights saved to {args.output_json}"
    )
    if args.torch_checkpoint:
        print(f"Torch checkpoint stored at {args.torch_checkpoint}")


if __name__ == "__main__":
    main()
