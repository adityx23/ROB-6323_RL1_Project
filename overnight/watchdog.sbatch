#!/usr/bin/env bash
# At 07:00 stop any pi0.5 training still running, so its eval (afterany) starts on the latest checkpoint.
#SBATCH --job-name=ovn-watchdog
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
echo "watchdog $(date -Is): stopping pi0.5 training jobs $JOBS"
for j in $JOBS; do squeue -j $j -h -o "%i %T"; scancel $j; done
