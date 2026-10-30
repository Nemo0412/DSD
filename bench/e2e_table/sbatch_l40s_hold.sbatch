#!/bin/bash
#SBATCH --job-name=l40s_hold
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err

# Holds gl001-003 so the A100 job can srun drafts with --overlap.
sleep 43000
