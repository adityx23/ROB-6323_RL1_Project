#!/usr/bin/env bash
# CPU-only: compute + cache pi0.5 norm stats for local/ocb_stack_3k with the ladder config (H=30 delta actions).
# -> runs/ladder/assets/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json (reused by smoke + full training).
#SBATCH --job-name=ldr-norm
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
export JAX_PLATFORMS=cpu
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
cd $PROJ/ladder
LDR_WORKERS=24 $PY pi05_ladder.py norm
cat $RLFT_ROOT/runs/ladder/assets/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json | head -c 600; echo
echo "end=$(date -Is)"
