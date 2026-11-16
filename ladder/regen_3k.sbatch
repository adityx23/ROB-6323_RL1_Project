#!/usr/bin/env bash
# OCBench block-double-task2 (cube stacking) visual demos at image interval 5, scaled up to ~3.3k successful.
# Existing parts p000-p003 (seeds 1-4, overnight run) are kept; this adds p004-p033 (seeds 5-34),
# 2 parts x 110 episodes per array task (same per-job load as the overnight regen that worked: ~1 h, ~85% success).
#   sbatch --array=0-14%2 regen_3k.sbatch
#SBATCH --job-name=ldr-regen
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
cd $RLFT_ROOT
run=$PROJ/torch-setup/run.sh
OUT=$RLFT_ROOT/data/ocbench/regen/visual-block-double-task2-v0/i5
echo "start=$(date -Is) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SLURM_CPUS_PER_TASK=8
P=()
for k in 0 1; do
  p=$((4 + 2 * SLURM_ARRAY_TASK_ID + k))
  name=$(printf 'train-p%03d' $p)
  if [ -f "$OUT/$name.npz" ]; then echo "$name exists, skipping"; continue; fi
  bash $run python -X faulthandler scripts/gen_data.py regen --task block-double-task2 --visual --episodes ${EPS:-110} \
      --interval 5 --max-parallel ${MAXPAR:-55} --seed $((p + 1)) --name $name --root $RLFT_ROOT/data/ocbench \
      > logs/ldr-regen-$name-${SLURM_ARRAY_JOB_ID}.log 2>&1 &
  P+=($!)
done
RC=0; for pid in "${P[@]}"; do wait $pid; r=$?; echo "part pid $pid exit=$r"; [ $r = 0 ] || RC=1; done
grep -h '"success_rate"' logs/ldr-regen-train-p*-${SLURM_ARRAY_JOB_ID}.log 2>/dev/null
echo "end=$(date -Is) rc=$RC"
exit $RC
