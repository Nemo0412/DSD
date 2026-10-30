#!/bin/bash
set -euo pipefail
for _ in $(seq 1 180); do
  [[ -f "${OUTDIR}/server.ip" ]] && break
  sleep 1
done
if [[ ! -f "${OUTDIR}/server.ip" ]]; then
  echo "server ip file missing" >&2
  exit 1
fi
IP=$(cat "${OUTDIR}/server.ip")
{
  echo "client $(hostname) server=${IP} port=${PORT}"
  date
  nvidia-smi -L || true
  ip -br addr || true
  echo "===== ping ====="
  ping -c 20 -i 0.2 "${IP}" || true
} > "${OUTDIR}/l40s_env.txt"
CSV="${OUTDIR}/trials.csv"
: > "${CSV}"
sleep 2
for run in $(seq 1 "${RUNS}"); do
  echo "===== cross run ${run}/${RUNS} $(date) ====="
  /share/apps/apptainer/bin/apptainer exec --nv "${IMG}" "${BIN}" \
    --role client --host "${IP}" --port "${PORT}" --gpu 0 \
    --out "${CSV}" --run "${run}" --trials "${TRIALS}"
done
echo "client done"
