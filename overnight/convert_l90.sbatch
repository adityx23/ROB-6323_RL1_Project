#!/usr/bin/env bash
#SBATCH --job-name=ovn-conv-l90
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
cd $OVN && $PY convert_libero90.py
