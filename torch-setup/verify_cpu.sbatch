#!/usr/bin/env bash
#SBATCH --job-name=rlft-verify-cpu
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
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
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
printf '%s\n' 'inputs=synthetic toy demonstrations; seed=0; no external data' \
  'output=/scratch/aap9348/rlft-bench/runs/verification/cpu-JOBID' \
  'command=bash /home/aap9348/rlft-bench-torch/run.sh python /home/aap9348/rlft-bench-torch/verify_cpu.py'
bash /home/aap9348/rlft-bench-torch/run.sh python /home/aap9348/rlft-bench-torch/verify_cpu.py
printf 'end=%s\n' "$(date --iso-8601=seconds)"
