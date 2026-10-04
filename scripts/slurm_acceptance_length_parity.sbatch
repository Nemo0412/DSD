#!/bin/bash
#SBATCH --job-name=dsd-acceptance-length-parity
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yf3005/dsd-real/runs/acceptance_length_parity_slurm_%j.log

set -euo pipefail
cd /scratch/yf3005/dsd-real
source scripts/env.sh
export HF_HUB_OFFLINE=1

READY_FILE=$(mktemp -u /scratch/yf3005/dsd-real/runs/draft_ready.XXXXXX)
rm -f "$READY_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_draft_server.py \
    --bind "tcp://127.0.0.1:5561" --ready-file "$READY_FILE" &
DRAFT_PID=$!
trap 'kill "$DRAFT_PID" 2>/dev/null || true; rm -f "$READY_FILE"' EXIT

READY_TIMEOUT_S=300
deadline=$((SECONDS + READY_TIMEOUT_S))
until [ -f "$READY_FILE" ]; do
    if ! kill -0 "$DRAFT_PID" 2>/dev/null; then
        echo "draft server died before becoming ready" >&2
        exit 1
    fi
    if [ "$SECONDS" -gt "$deadline" ]; then
        echo "draft server did not become ready within ${READY_TIMEOUT_S}s" >&2
        exit 1
    fi
    sleep 2
done
DRAFT_ADDR=$(cat "$READY_FILE")
echo "draft server ready at $DRAFT_ADDR"

# GPU1 hosts both legs of this script (fused's in-process draft + target,
# then distributed's target only) -- the fused leg needs the draft model
# resident locally regardless, and 2a's own gpu_frac split already proves
# target+draft coexist fine on one device.
CUDA_VISIBLE_DEVICES=1 python scripts/verify_acceptance_length_parity.py \
    --draft-addr "$DRAFT_ADDR"
