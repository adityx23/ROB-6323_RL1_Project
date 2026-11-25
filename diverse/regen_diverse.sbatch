#!/usr/bin/env bash
# Diverse OCBench cube-stacking demos (5 colors with stacking rules, colored prompts, wider starts/speeds, light
# DART noise): 2 parts x EPS episodes per array task, part p = 2*task + k, seed 101 + p. Skips existing parts.
#   ./submit.sh --array=0-0 --export=ALL,EPS=110,NPARTS=1 diverse/regen_diverse.sbatch     # pilot (1 part)
#   ./submit.sh --array=0-6 diverse/regen_diverse.sbatch                                   # full set
#SBATCH --job-name=dv-regen
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
OUT=$RLFT_ROOT/data/ocbench/regen-diverse/visual-block-double-task2-v0/i5
echo "start=$(date -Is) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SLURM_CPUS_PER_TASK=8
P=()
for k in $(seq 0 $((${NPARTS:-2} - 1))); do
  p=$((2 * SLURM_ARRAY_TASK_ID + k))
  name=$(printf 'train-p%03d' $p)
  if [ -f "$OUT/$name.npz" ]; then echo "$name exists, skipping"; continue; fi
  bash $PROJ/torch-setup/run.sh python -X faulthandler $PROJ/diverse/gen_colored.py --episodes ${EPS:-110} \
      --seed $((101 + p)) --name $name --root $RLFT_ROOT/data/ocbench \
      > $RLFT_ROOT/logs/dv-regen-$name-${SLURM_ARRAY_JOB_ID}.log 2>&1 &
  P+=($!)
done
RC=0; for pid in "${P[@]}"; do wait $pid; r=$?; echo "part pid $pid exit=$r"; [ $r = 0 ] || RC=1; done
for k in $(seq 0 $((${NPARTS:-2} - 1))); do
  name=$(printf 'train-p%03d' $((2 * SLURM_ARRAY_TASK_ID + k)))
  [ -f "$OUT/$name.npz" ] && bash $PROJ/torch-setup/run.sh python $PROJ/diverse/inspect_part.py $OUT/$name.npz \
      ${SHEET:+--sheet $RLFT_ROOT/logs/dv-sheet-$name.png}
done
echo "end=$(date -Is) rc=$RC"
exit $RC
