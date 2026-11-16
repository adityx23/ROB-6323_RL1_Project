#!/usr/bin/env bash
#SBATCH --job-name=ovn-regen-dbg
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -uo pipefail
cd /scratch/aap9348/rlft-bench
echo "start=$(date -Is) host=$(hostname)"
bash /home/aap9348/rlft-bench-torch/run.sh python -X faulthandler scripts/gen_data.py regen --task block-double-task2 --visual \
  --episodes 4 --interval 5 --max-parallel 4 --seed 99 --name dbg --root /scratch/aap9348/rlft-bench/data/ocbench 2>&1
echo "exit=$? end=$(date -Is)"
ls -la data/ocbench/regen/*/i5/ 2>&1
