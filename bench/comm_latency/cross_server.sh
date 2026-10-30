#!/bin/bash
set -euo pipefail
pick_ip() {
  local ip=""
  for nic in ib0 ib1 ibs0 ibp0s0; do
    ip=$(ip -4 -o addr show "$nic" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return
    fi
  done
  hostname -I | awk '{print $1}'
}
IP=$(pick_ip)
{
  echo "server $(hostname) ip=${IP} port=${PORT}"
  date
  nvidia-smi -L || true
  ip -br addr || true
} > "${OUTDIR}/h100_env.txt"
echo "${IP}" > "${OUTDIR}/server.ip"
echo "server ready ${IP}:${PORT}"
for run in $(seq 1 "${RUNS}"); do
  echo "server run ${run} $(date)"
  /share/apps/apptainer/bin/apptainer exec --nv "${IMG}" "${BIN}" \
    --role server --bind "${IP}" --port "${PORT}" --gpu 0 \
    --out "${OUTDIR}/server.csv" --run "${run}" --trials "${TRIALS}"
done
echo "server done"
