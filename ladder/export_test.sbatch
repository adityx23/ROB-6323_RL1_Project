#!/usr/bin/env bash
# CPU-only: regression check of export_3k.py against local/ocb_stack, then a 30-episode timing export.
#SBATCH --job-name=ldr-export-test
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$RLFT_ROOT/cache/huggingface/lerobot
run=$PROJ/torch-setup/run.sh
bash $run python $PROJ/ladder/export_3k.py check --episodes 3
bash $run python $PROJ/ladder/export_3k.py export --num 30 --repo local/ocb_stack_exporttest
rm -rf $HF_LEROBOT_HOME/local/ocb_stack_exporttest
