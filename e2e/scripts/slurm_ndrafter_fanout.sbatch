#!/bin/bash
#SBATCH --job-name=dsd-ndrafter-fanout
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yf3005/dsd-real/runs/ndrafter_fanout_slurm_%j.log

set -euo pipefail
cd /scratch/yf3005/dsd-real
source scripts/env.sh
export HF_HUB_OFFLINE=1

# Two drafters, same tiny model, sharing GPU0 (same coexistence trick 2a
# uses for target+draft: gpu_memory_utilization fractions, not separate
# devices) -- the point of this gate is routing/fan-out, not GPU count.
READY0=$(mktemp -u /scratch/yf3005/dsd-real/runs/draft_ready.XXXXXX); rm -f "$READY0"
READY1=$(mktemp -u /scratch/yf3005/dsd-real/runs/draft_ready.XXXXXX); rm -f "$READY1"

CUDA_VISIBLE_DEVICES=0 python scripts/run_draft_server.py \
    --bind "tcp://127.0.0.1:5570" --drafter-id d0 --gpu-frac 0.35 --ready-file "$READY0" &
D0_PID=$!
CUDA_VISIBLE_DEVICES=0 python scripts/run_draft_server.py \
    --bind "tcp://127.0.0.1:5571" --drafter-id d1 --gpu-frac 0.35 --ready-file "$READY1" &
D1_PID=$!
trap 'kill "$D0_PID" "$D1_PID" 2>/dev/null || true; rm -f "$READY0" "$READY1"' EXIT

READY_TIMEOUT_S=300
deadline=$((SECONDS + READY_TIMEOUT_S))
until [ -f "$READY0" ] && [ -f "$READY1" ]; do
    if ! kill -0 "$D0_PID" 2>/dev/null || ! kill -0 "$D1_PID" 2>/dev/null; then
        echo "a draft server died before becoming ready" >&2
        exit 1
    fi
    if [ "$SECONDS" -gt "$deadline" ]; then
        echo "draft servers did not become ready within ${READY_TIMEOUT_S}s" >&2
        exit 1
    fi
    sleep 2
done
D0_ADDR=$(cat "$READY0")
D1_ADDR=$(cat "$READY1")
echo "draft servers ready: d0=$D0_ADDR d1=$D1_ADDR"

CUDA_VISIBLE_DEVICES=1 python scripts/verify_ndrafter_fanout.py \
    --draft-addr "d0=$D0_ADDR" --draft-addr "d1=$D1_ADDR"
