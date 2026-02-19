#!/bin/bash
#SBATCH --job-name=qeig
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

# Sweep over N and potential:
# total tasks = len(Ns) * len(pots) = 4 * 2 = 8  -> array indices 0..7
Ns=(80)
pots=(well harmonic quartic)

task=${SLURM_ARRAY_TASK_ID}
n_idx=$(( task / ${#pots[@]} ))
p_idx=$(( task % ${#pots[@]} ))

N=${Ns[$n_idx]}
pot=${pots[$p_idx]}

# Always run from the repo root (where you submitted sbatch)
REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

mkdir -p logs results

echo "JobID=$SLURM_JOB_ID TaskID=$task Host=$(hostname) N=$N potential=$pot"
echo "Start: $(date)"

# Program stdout -> run log; time + stderr -> time file (and error messages if any)
time -p python src/eigen.py --N "$N" --potential "$pot" --neigs 10 --GS True \
  > "logs/run_N${N}_${pot}.log" 2> "logs/time_N${N}_${pot}.txt"

echo "End: $(date)"