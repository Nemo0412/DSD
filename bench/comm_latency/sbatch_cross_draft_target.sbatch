#!/bin/bash
#SBATCH --job-name=dt-cross
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=00:20:00
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
#SBATCH --time=00:20:00

set -euo pipefail
ROOT="/home/ll5914/DSD-MLsys2027"
export ROOT
export PY="/scratch/yf3005/miniconda3/envs/vllm027/bin/python"
export OUTDIR="${ROOT}/results/comm_latency/draft_target_cross_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}/ready" "${OUTDIR}/done"

srun --export=ALL --het-group=0 --ntasks=1 --cpus-per-task=4 \
  bash "${ROOT}/bench/comm_latency/cross_draft.sh" \
  > "${OUTDIR}/draft.log" 2>&1 &
DRAFT_PID=$!

srun --export=ALL --het-group=1 --ntasks=1 --cpus-per-task=4 \
  bash "${ROOT}/bench/comm_latency/cross_target.sh" \
  | tee "${OUTDIR}/target.log"

wait "${DRAFT_PID}" || true

python3 - << PY
import csv, statistics, collections
rows=list(csv.DictReader(open("${OUTDIR}/trials.csv")))
measured=[r for r in rows if r["phase"]!="warmup"]
groups=collections.defaultdict(list)
for r in measured:
    groups[r["phase"]].append(float(r["net_excess_s"]))
print("samples", len(measured))
for phase, vals in groups.items():
    print(f"{phase}: n={len(vals)} median_s={statistics.median(vals):.9f} "
          f"min_s={min(vals):.9f} max_s={max(vals):.9f}")
PY
echo "results ${OUTDIR}"
