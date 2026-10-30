#!/bin/bash
#SBATCH --job-name=e2e-remain
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err

# Account, reservation, nodes, and GPUs are set on the sbatch command.
# DRAFT_JOB is the L40S hold submitted first.
set -uo pipefail
export A100_JOB="${SLURM_JOB_ID}"
export DRAFT_JOB="${DRAFT_JOB:?set DRAFT_JOB to the L40S hold}"
export SKIP_WAIT=1
export RUN_MATRIX=1
export OUT16=/home/ll5914/DSD-MLsys2027/results/e2e_a100x4_l40sx12_19637436
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR"
bash /home/ll5914/DSD-MLsys2027/bench/e2e_table/run_followup.sh
