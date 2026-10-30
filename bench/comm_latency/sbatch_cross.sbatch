#!/bin/bash
#SBATCH --job-name=lat-cross
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH hetjob
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00

set -euo pipefail
ROOT="/home/ll5914/DSD-MLsys2027"
export BIN="${ROOT}/bench/comm_latency/net_latency"
export IMG="/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
export OUTDIR="${ROOT}/results/comm_latency/cross_${SLURM_JOB_ID}"
export PORT=$((20000 + SLURM_JOB_ID % 20000))
export RUNS="${RUNS:-5}"
export TRIALS="${TRIALS:-5}"
mkdir -p "${OUTDIR}"

srun --export=ALL --het-group=0 --ntasks=1 --cpus-per-task=4 \
  bash "${ROOT}/bench/comm_latency/cross_server.sh" \
  > "${OUTDIR}/server.log" 2>&1 &
SERVER_PID=$!

srun --export=ALL --het-group=1 --ntasks=1 --cpus-per-task=4 \
  bash "${ROOT}/bench/comm_latency/cross_client.sh" \
  | tee "${OUTDIR}/client.log"

wait "${SERVER_PID}"

python3 - << PY
import csv, statistics, collections
path = "${OUTDIR}/trials.csv"
rows = list(csv.DictReader(open(path)))
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["kind"], int(r["size_bytes"]))].append(float(r["rtt_us"]))
print(f"{'kind':<14} {'bytes':>10} {'n':>4} values")
for (kind, nbytes), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    shown = " ".join(f"{v:.3f}" for v in vals)
    print(f"{kind:<14} {nbytes:10d} {len(vals):4d} median={statistics.median(vals):.3f}  [{shown}]")
PY
echo "results ${OUTDIR}"
