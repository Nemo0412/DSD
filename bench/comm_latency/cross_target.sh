#!/bin/bash
set -euo pipefail
for _ in $(seq 1 120); do
  [[ -f "${OUTDIR}/ready" ]] && break
  sleep 0.5
done
RAW=$(cat "${OUTDIR}/ready")
PORT=${RAW##*:}
ADDR=$(python3 - << PY
import socket, sys
port = int("${PORT}")
ips = []
for line in open("${OUTDIR}/draft_env.txt"):
    for tok in line.split():
        if tok.count(".") == 3 and tok.split(".")[0].isdigit():
            ip = tok.split("/")[0]
            if ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
ok = None
for ip in ips:
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect((ip, port))
        ok = ip
        print(f"reachable {ip}:{port}", file=sys.stderr, flush=True)
        break
    except OSError as exc:
        print(f"unreachable {ip}:{port} {exc}", file=sys.stderr, flush=True)
    finally:
        s.close()
if ok is None:
    raise SystemExit("no reachable draft address")
print(f"tcp://{ok}:{port}")
PY
)
ADDR=$(echo "${ADDR}" | tail -1)
{
  echo "target $(hostname) addr=${ADDR}"
  date
  ip -br addr || true
} > "${OUTDIR}/target_env.txt"
export PYTHONPATH="${ROOT}"
"${PY}" "${ROOT}/bench/comm_latency/draft_target_comm.py" \
  --role client --addr "${ADDR}" --out "${OUTDIR}/trials.csv" \
  --runs 5 --steps 20
touch "${OUTDIR}/done"
