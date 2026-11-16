#!/usr/bin/env bash
# CPU job: upload the ladder checkpoints (params + assets) to a public HF repo. Resumable; rerun if it stops.
#SBATCH --job-name=ldr-upload
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
export HF_TOKEN=$(cat /home/aap9348/.config/ldr-hf/token) HF_HUB_OFFLINE=0
export HF_HOME=/scratch/aap9348/rlft-bench/cache/hf-upload HF_HUB_ENABLE_HF_TRANSFER=0
echo "start=$(date -Is) host=$(hostname)"
/scratch/aap9348/rlft-bench/.venv/bin/python -I /home/aap9348/rlft-bench-torch/ladder/upload_hf.py "$@"
echo "end=$(date -Is)"
