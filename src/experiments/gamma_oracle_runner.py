import argparse
import copy
import json
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-policy gamma oracle sweeps")
    parser.add_argument("--config", required=True, help="Base simulator config YAML")
    parser.add_argument("--scenario-id", required=True, help="Label for this sweep (e.g., gsm8k_rtt1x)")
    parser.add_argument(
        "--gammas",
        default="1,2,3,4,5,6,7,8",
        help="Comma separated list of gamma candidates. Use 1 for fused mode",
    )
    parser.add_argument(
        "--drafter-counts",
        default="",
        help="Comma separated list of drafter counts to sweep (overrides config). Empty = use config value",
    )
    parser.add_argument(
        "--rtt-multipliers",
        default="",
        help="Comma list of RTT multipliers (e.g., 1.0,1.2). Empty = use config latency",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for all runs")
    parser.add_argument(
        "--output-dir",
        default="data/gamma_oracle",
        help="Directory to store raw logs and aggregated dataset",
    )
    parser.add_argument(
        "--distributed-mode",
        default="hybrid",
        choices=["distributed", "hybrid"],
        help="Execution mode for distributed candidates (defaults to hybrid)",
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        default=None,
        help="Optional TTFT SLO threshold when picking the oracle action",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=None,
        help="Optional TPOT SLO threshold (applied to max tpot sample)",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only aggregate existing logs (skips launching simulator)",
    )
    parser.add_argument(
        "--sim-binary",
        default=None,
        help="Path to sim.py; defaults to <repo_root>/src/sim.py",
    )
    parser.add_argument(
        "--extra-metadata",
        default="{}",
        help="JSON string merged into run_metadata for every candidate",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=200,
        help="Stop each simulation after this many completed conversations (default: 200)",
    )
    return parser.parse_args()


def _normalize_gamma_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one gamma candidate is required")
    return values


def _normalize_float_list(raw: str) -> List[float]:
    values: List[float] = []
    if not raw:
        return values
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _normalize_int_list(raw: str) -> List[int]:
    values: List[int] = []
    if not raw:
        return values
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _write_temp_config(cfg: Dict, tmp_dir: Path, gamma: int) -> Path:
    path = tmp_dir / f"gamma_{gamma}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _apply_drafter_count(cfg: Dict[str, Any], drafter_count: int) -> None:
    if drafter_count is None:
        return
    auto_topology = cfg.get("auto_topology") or {}
    clusters = auto_topology.get("clusters") or []
    updated = False
    for cluster in clusters:
        drafts_cfg = cluster.get("drafts")
        if isinstance(drafts_cfg, dict) and "count" in drafts_cfg:
            drafts_cfg["count"] = int(drafter_count)
            updated = True
    if updated:
        return
    devices = cfg.get("devices") or []
    for device in devices:
        if device.get("role", "draft").lower() == "draft":
            device["count"] = int(drafter_count)
            updated = True
    if updated:
        return
    raise ValueError(
        "Config does not expose drafter counts; expected auto_topology.clusters[].drafts.count or devices[]."
    )


def _apply_rtt_multiplier(cfg: Dict[str, Any], multiplier: float) -> None:
    if multiplier is None:
        return
    net_cfg = cfg.setdefault("network_config", {})
    base_rtt = float(net_cfg.get("base_rtt_ms", 5.0))
    net_cfg["base_rtt_ms"] = base_rtt * multiplier


def _run_sim(sim_path: Path, config_path: Path, repo_root: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing
        else f"{repo_root}:{existing}"
    )
    cmd = [sys.executable, str(sim_path), "--config", str(config_path)]
    subprocess.run(cmd, check=True, env=env)


def _load_records(paths: List[Path]) -> Dict[str, Dict[int, Dict]]:
    result: Dict[str, Dict[int, Dict]] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                request_id = rec.get("request_id")
                run_meta = rec.get("run_metadata", {})
                gamma = int(run_meta.get("gamma_candidate", rec.get("gamma", 0)))
                if request_id is None:
                    continue
                result.setdefault(request_id, {})[gamma] = rec
    return result


def _passes_slo(rec: Dict, ttft_slo: float | None, tpot_slo: float | None) -> bool:
    metrics = rec.get("metrics") or {}
    if ttft_slo is not None:
        ttft = metrics.get("ttft_ms")
        if ttft is None:
            return False
        if float(ttft) > ttft_slo:
            return False
    if tpot_slo is not None:
        samples = metrics.get("tpot_samples") or []
        if not samples:
            return False
        if max(float(s) for s in samples) > tpot_slo:
            return False
    return True


def _pick_best(
    candidates: Dict[int, Dict], ttft_slo: float | None, tpot_slo: float | None
) -> Tuple[int, Dict] | None:
    best_gamma = None
    best_rec: Dict | None = None
    best_duration = float("inf")
    for gamma, rec in candidates.items():
        if not _passes_slo(rec, ttft_slo, tpot_slo):
            continue
        metrics = rec.get("metrics") or {}
        duration = float(metrics.get("duration_ms", float("inf")))
        if duration < best_duration:
            best_duration = duration
            best_gamma = gamma
            best_rec = rec
    if best_gamma is None or best_rec is None:
        return None
    return best_gamma, best_rec


def _aggregate(  # pylint: disable=too-many-arguments
    raw_paths: List[Path],
    dataset_path: Path,
    ttft_slo: float | None,
    tpot_slo: float | None,
) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_records(raw_paths)
    kept = 0
    with dataset_path.open("w", encoding="utf-8") as out:
        for request_id, candidates in records.items():
            best = _pick_best(candidates, ttft_slo, tpot_slo)
            if best is None:
                continue
            best_gamma, best_rec = best
            features = best_rec.get("features")
            entry = {
                "request_id": request_id,
                "best_gamma": best_gamma,
                "best_mode": best_rec.get("mode"),
                "features": features,
                "metrics": best_rec.get("metrics"),
                "run_metadata": best_rec.get("run_metadata", {}),
                "candidates": {
                    gamma: {
                        "mode": rec.get("mode"),
                        "metrics": rec.get("metrics"),
                    }
                    for gamma, rec in candidates.items()
                },
            }
            out.write(json.dumps(entry, ensure_ascii=False))
            out.write("\n")
            kept += 1
    print(f"Wrote {kept} oracle rows to {dataset_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sim_path = Path(args.sim_binary) if args.sim_binary else repo_root / "src" / "sim.py"
    base_cfg = yaml.safe_load(Path(args.config).read_text())
    gammas = _normalize_gamma_list(args.gammas)
    drafter_counts = _normalize_int_list(args.drafter_counts)
    rtt_mults = _normalize_float_list(args.rtt_multipliers)
    if not drafter_counts:
        drafter_counts = [None]
    if not rtt_mults:
        rtt_mults = [None]
    output_root = Path(args.output_dir)
    raw_dir = output_root / "raw" / args.scenario_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    extra_metadata = json.loads(args.extra_metadata)

    if not args.skip_runs:
        tmp_dir = Path(tempfile.mkdtemp(prefix="oracle_cfg_"))
        try:
            for drafter_count in drafter_counts:
                for rtt_mult in rtt_mults:
                    scenario_suffix = []
                    if drafter_count is not None:
                        scenario_suffix.append(f"d{drafter_count}")
                    if rtt_mult is not None:
                        scenario_suffix.append(f"r{rtt_mult:.2f}")
                    scenario_label = "_".join(filter(None, [args.scenario_id] + scenario_suffix))
                    for gamma in gammas:
                        cfg = copy.deepcopy(base_cfg)
                        if args.seed is not None:
                            cfg["seed"] = int(args.seed)
                        cfg["max_conversations"] = int(args.max_conversations)
                        if drafter_count is not None:
                            _apply_drafter_count(cfg, drafter_count)
                        if rtt_mult is not None:
                            _apply_rtt_multiplier(cfg, rtt_mult)
                        mode = "fused" if gamma == 1 else args.distributed_mode
                        spec_cfg = cfg.setdefault("speculation", {})
                        spec_cfg["execution_mode"] = mode
                        cfg["speculation_execution_mode"] = mode
                        cfg["gamma"] = int(gamma)
                        spec_cfg["gamma_policy"] = {"type": "constant", "default_gamma": int(gamma)}
                        spec_cfg.setdefault("acceptance", {})["disable_model"] = True
                        perf_cfg = cfg.setdefault("performance_model", {})
                        vidur_cfg = perf_cfg.setdefault("vidur", {})
                        vidur_cfg["realtime_enabled"] = True
                        oracle_cfg = spec_cfg.setdefault("gamma_oracle_logging", {})
                        raw_path = raw_dir / f"scenario_{scenario_label}_gamma{gamma}.jsonl"
                        oracle_cfg["enabled"] = True
                        oracle_cfg["output_path"] = str(raw_path)
                        run_metadata = {
                            "scenario_id": scenario_label,
                            "gamma_candidate": int(gamma),
                            "mode": mode,
                        }
                        if drafter_count is not None:
                            run_metadata["drafter_count"] = int(drafter_count)
                        if rtt_mult is not None:
                            run_metadata["rtt_multiplier"] = float(rtt_mult)
                        if args.seed is not None:
                            run_metadata["seed"] = int(args.seed)
                        run_metadata.update(extra_metadata or {})
                        oracle_cfg["run_metadata"] = run_metadata
                        cfg_path = _write_temp_config(cfg, tmp_dir, gamma)
                        print(
                            f"Running scenario={scenario_label} gamma={gamma} (mode={mode}) -> {raw_path}"
                        )
                        _run_sim(sim_path, cfg_path, repo_root)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    raw_paths = sorted(raw_dir.glob("scenario_*_gamma*.jsonl"))
    if not raw_paths:
        raise SystemExit(f"No raw gamma oracle logs found in {raw_dir}")
    dataset_path = output_root / "dataset" / f"{args.scenario_id}.jsonl"
    _aggregate(raw_paths, dataset_path, args.ttft_slo_ms, args.tpot_slo_ms)


if __name__ == "__main__":
    main()
