#!/usr/bin/env bash
# ACT, shared recipe. sbatch --export=ALL,BENCH=ocbs train_act.sbatch
#SBATCH --job-name=ovn-act
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID bench=$BENCH"
cd $OVN
$PY act_train.py --bench $BENCH --out $RUNS/act_$BENCH --max-hours ${MAX_HOURS:-7.0}
echo "end=$(date -Is)"
