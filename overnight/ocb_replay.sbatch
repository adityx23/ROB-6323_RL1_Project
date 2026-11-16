#!/usr/bin/env bash
#SBATCH --job-name=ovn-replay
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
source /home/aap9348/rlft-bench-torch/overnight/env.sh
cd /scratch/aap9348/rlft-bench
OMP_NUM_THREADS=1 PYTHONPATH=/scratch/aap9348/rlft-bench $PY $OVN/ocb_replay.py 2>&1 | grep -vE 'Warning|solver iterations|warn_overflow|Module .* load'
