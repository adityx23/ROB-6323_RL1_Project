#!/usr/bin/env bash
#SBATCH --job-name=ovn-conv-l90
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
source /home/aap9348/rlft-bench-torch/overnight/env.sh
cd $OVN && $PY convert_libero90.py
