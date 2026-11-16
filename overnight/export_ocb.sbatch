#!/usr/bin/env bash
#SBATCH --job-name=ovn-export-ocb
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
cd /scratch/aap9348/rlft-bench
bash /home/aap9348/rlft-bench-torch/run.sh python /home/aap9348/rlft-bench-torch/overnight/export_ocb_lerobot.py
