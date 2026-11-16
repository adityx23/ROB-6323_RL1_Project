#!/usr/bin/env bash
# ACT, shared recipe. sbatch --export=ALL,BENCH=l90|ocbs train_act.sbatch
#SBATCH --job-name=ovn-act
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
source /home/aap9348/rlft-bench-torch/overnight/env.sh
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID bench=$BENCH"
cd $OVN
$PY act_train.py --bench $BENCH --out $RUNS/act_$BENCH --max-hours ${MAX_HOURS:-7.0}
echo "end=$(date -Is)"
