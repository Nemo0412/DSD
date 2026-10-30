#!/bin/bash
# Intra-node GPU-GPU latency. Expects 2 GPUs on this node.
set -euo pipefail

ROOT="/home/ll5914/DSD-MLsys2027"
BIN="${ROOT}/bench/comm_latency/p2p_latency"
IMG="/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
TAG="${1:?tag required, e.g. h100 or l40s}"
OUTDIR="${ROOT}/results/comm_latency/${TAG}_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUTDIR}"

{
  echo "===== hostname ====="
  hostname
  date
  echo "===== nvidia-smi -L ====="
  nvidia-smi -L
  echo "===== nvidia-smi topo ====="
  nvidia-smi topo -m || true
  echo "===== nvidia-smi nvlink ====="
  nvidia-smi nvlink --status || true
} > "${OUTDIR}/topo.txt" 2>&1

CSV="${OUTDIR}/trials.csv"
: > "${CSV}"

RUNS="${RUNS:-5}"
TRIALS="${TRIALS:-5}"
for run in $(seq 1 "${RUNS}"); do
  echo "===== ${TAG} run ${run}/${RUNS} $(date) ====="
  /share/apps/apptainer/bin/apptainer exec --nv "${IMG}" "${BIN}" \
    --out "${CSV}" --run "${run}" --trials "${TRIALS}" --gpus 0,1 \
    | tee -a "${OUTDIR}/run.log"
done

echo "results ${OUTDIR}"
python3 - << PY
import csv, statistics, collections
path = "${CSV}"
rows = list(csv.DictReader(open(path)))
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["kind"], int(r["size_bytes"]))].append(float(r["latency_us"]))
print(f"{'kind':<16} {'bytes':>10} {'n':>4} {'each_us'}")
for (kind, nbytes), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    shown = " ".join(f"{v:.3f}" for v in vals)
    med = statistics.median(vals)
    print(f"{kind:<16} {nbytes:10d} {len(vals):4d} median={med:.3f}  [{shown}]")
PY
echo "wrote ${CSV}"
