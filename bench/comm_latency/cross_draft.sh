#!/bin/bash
set -euo pipefail
pick_ip() {
  local ip=""
  for nic in $(ls /sys/class/net | grep -E '^(ibs|ib)'); do
    ip=$(ip -4 -o addr show "$nic" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return
    fi
  done
  hostname -I | awk '{print $1}'
}
{
  echo "draft $(hostname)"
  date
  ip -br addr || true
} > "${OUTDIR}/draft_env.txt"
export PYTHONPATH="${ROOT}"
# Bind every interface. The target probes which address it can actually reach
# (H100 nodes have several IB rails; only some are routed to L40S).
"${PY}" "${ROOT}/bench/comm_latency/draft_target_comm.py" \
  --role server --bind-host 0.0.0.0 --advertise-host 0.0.0.0 \
  --ready-file "${OUTDIR}/ready" --done-file "${OUTDIR}/done"
