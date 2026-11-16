#!/usr/bin/env bash
#SBATCH --job-name=rlft-verify-cpu-retry
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
printf 'job_id=%s\njob_name=%s\naccount=%s\nhost=%s\nnodes=%s\nstart=%s\nworking_directory=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "${SLURM_JOB_ACCOUNT:-unknown}" "$(hostname)" \
  "$SLURM_JOB_NODELIST" "$(date --iso-8601=seconds)" "$PWD"
module load anaconda3/2025.06
module list
git rev-parse HEAD
git -C ../lair-openpi rev-parse HEAD
sha256sum uv.lock
myquota
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export RLFT_REUSE_BASE=$RLFT_ROOT/runs/verification/cpu-21531365/base
printf '%s\n' 'inputs=verified synthetic toy base from job 21531365; seed=0; no external data' \
  'output=$RLFT_ROOT/runs/verification/cpu-JOBID' \
  'command=bash $PROJ/torch-setup/run.sh python $PROJ/torch-setup/verify_cpu.py'
bash $PROJ/torch-setup/run.sh python $PROJ/torch-setup/verify_cpu.py
printf 'end=%s\n' "$(date --iso-8601=seconds)"
