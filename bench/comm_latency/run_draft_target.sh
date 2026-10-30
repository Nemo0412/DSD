#!/bin/bash
# Same-node draft process + target process. ZMQ over 127.0.0.1, matching the
# A6000 sbatch (draft on one GPU process, target on another, tcp://127.0.0.1).
set -euo pipefail
ROOT="/home/ll5914/DSD-MLsys2027"
PY="/scratch/yf3005/miniconda3/envs/vllm027/bin/python"
TAG="${1:?tag}"
OUTDIR="${ROOT}/results/comm_latency/draft_target_${TAG}_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUTDIR}"
READY="${OUTDIR}/ready"
DONE="${OUTDIR}/done"
rm -f "${READY}" "${DONE}"
export PYTHONPATH="${ROOT}"

hostname > "${OUTDIR}/host.txt"
date >> "${OUTDIR}/host.txt"

"${PY}" "${ROOT}/bench/comm_latency/draft_target_comm.py" \
  --role server --bind-host 127.0.0.1 --advertise-host 127.0.0.1 \
  --ready-file "${READY}" --done-file "${DONE}" \
  > "${OUTDIR}/draft.log" 2>&1 &
DRAFT_PID=$!
for _ in $(seq 1 60); do
  [[ -f "${READY}" ]] && break
  if ! kill -0 "${DRAFT_PID}" 2>/dev/null; then
    echo "draft exited early" >&2
    cat "${OUTDIR}/draft.log" >&2
    exit 1
  fi
  sleep 0.2
done
ADDR=$(cat "${READY}")
echo "target -> ${ADDR}"
"${PY}" "${ROOT}/bench/comm_latency/draft_target_comm.py" \
  --role client --addr "${ADDR}" --out "${OUTDIR}/trials.csv" \
  --runs 5 --steps 20 | tee "${OUTDIR}/target.log"
touch "${DONE}"
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
