#!/usr/bin/env bash
#SBATCH --job-name=ovn-replay
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
source $PROJ/overnight/env.sh
cd $RLFT_ROOT
OMP_NUM_THREADS=1 PYTHONPATH=$RLFT_ROOT $PY $OVN/ocb_replay.py 2>&1 | grep -vE 'Warning|solver iterations|warn_overflow|Module .* load'
