#!/usr/bin/env bash
# OCBench block-double-task2 (cube stacking) visual demos at image interval 5: 4 parts x 65 episodes,
# then export 200 successful ones to LeRobot (local/ocb_stack).
#SBATCH --job-name=ovn-regen
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -uo pipefail
cd /scratch/aap9348/rlft-bench
run=/home/aap9348/rlft-bench-torch/run.sh
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
export SLURM_CPUS_PER_TASK=8
P=()
for p in ${PARTS:-0 1 2 3}; do
  bash $run python -X faulthandler scripts/gen_data.py regen --task block-double-task2 --visual --episodes ${EPS:-110} \
      --interval 5 --max-parallel ${MAXPAR:-55} --seed $((p+1)) --name train-p00$p --root /scratch/aap9348/rlft-bench/data/ocbench \
      > logs/ovn-regen-p$p-$SLURM_JOB_ID.log 2>&1 &
  P+=($!)
done
RC=0; for pid in "${P[@]}"; do wait $pid; r=$?; echo "part pid $pid exit=$r"; [ $r = 0 ] || RC=1; done; [ $RC = 0 ] || exit 1
echo "regen done $(date -Is)"; ls -la data/ocbench/regen/visual-block-double-task2-v0/i5/
[ "${NOEXPORT:-0}" = 1 ] && exit 0
bash $run python /home/aap9348/rlft-bench-torch/overnight/export_ocb_lerobot.py || exit 1
echo "end=$(date -Is)"
