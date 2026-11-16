#!/usr/bin/env bash
#SBATCH --job-name=ovn-ltime
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
source $PROJ/overnight/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /tmp; echo "host=$(hostname) $(date -Is)"; df -h /tmp | tail -1
PYTHONPATH=$LIBERO_SRC $LIBERO_PY $OVN/libero_timing.py soft 2>&1 | grep -E '^(import|soft|hard|reset|100)|Error'
PYTHONPATH=$LIBERO_SRC $LIBERO_PY $OVN/libero_timing.py hard 2>&1 | grep -E '^(import|soft|hard|reset|100)|Error'
