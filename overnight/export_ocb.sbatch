#!/usr/bin/env bash
#SBATCH --job-name=ovn-export-ocb
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd $RLFT_ROOT
bash $PROJ/torch-setup/run.sh python $PROJ/overnight/export_ocb_lerobot.py
