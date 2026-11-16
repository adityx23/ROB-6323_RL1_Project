#!/usr/bin/env bash
#SBATCH --job-name=rlft-verify-gpu
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err

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
unset JAX_PLATFORMS
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
printf '%s\n' 'inputs=installed simulator assets; generated oracle data; seed=0; no dataset/model downloads' \
  'output=/scratch/aap9348/rlft-bench/runs/verification/gpu-JOBID' \
  'command=bash /home/aap9348/rlft-bench-torch/run.sh python /home/aap9348/rlft-bench-torch/verify_gpu.py'
bash /home/aap9348/rlft-bench-torch/run.sh python /home/aap9348/rlft-bench-torch/verify_gpu.py
printf 'end=%s\n' "$(date --iso-8601=seconds)"
