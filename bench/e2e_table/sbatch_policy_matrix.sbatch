#!/bin/bash
#SBATCH --job-name=e2e-matrix
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err

# Account, reservation, node, and GPUs are set on the sbatch command.
# Runs the 16-GPU routing x batching matrix inside this A100 allocation.
# Drafts use the already-running L40S hold in /scratch/ll5914/tmp/draft_job_id.
set -uo pipefail
export A100_JOB="${SLURM_JOB_ID}"
export DRAFT_JOB="$(tr -d '[:space:]' < /scratch/ll5914/tmp/draft_job_id)"
export INSIDE_ALLOC=1
export N_TARGETS=4
export N_DRAFTERS=12
export N_DRAFT_NODES=3
export TMPDIR=/scratch/ll5914/tmp
export ROOT=/home/ll5914/DSD-MLsys2027
mkdir -p "$TMPDIR" "$ROOT/logs"
echo "A100_JOB=$A100_JOB DRAFT_JOB=$DRAFT_JOB host=$(hostname) start=$(date -Is)"
if ! squeue -j "$DRAFT_JOB" -h -t R | grep -q .; then
  echo "DRAFT_JOB $DRAFT_JOB is not running"
  exit 1
fi
for h in ga001 gl001 gl002 gl003; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    "bash /scratch/ll5914/tmp/stop_keepalive.sh; bash /scratch/ll5914/tmp/stop_manual_drafts.sh" \
    || echo "cleanup ssh $h failed"
done
sleep 3
bash "$ROOT/bench/e2e_table/run_policy_matrix.sh"
echo "matrix script exited $? at $(date -Is)"
