#!/usr/bin/env bash
# CPU-only: regression check of export_3k.py against local/ocb_stack, then a 30-episode timing export.
#SBATCH --job-name=ldr-export-test
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=/scratch/aap9348/rlft-bench/cache/huggingface/lerobot
run=/home/aap9348/rlft-bench-torch/run.sh
bash $run python /home/aap9348/rlft-bench-torch/ladder/export_3k.py check --episodes 3
bash $run python /home/aap9348/rlft-bench-torch/ladder/export_3k.py export --num 30 --repo local/ocb_stack_exporttest
rm -rf $HF_LEROBOT_HOME/local/ocb_stack_exporttest
