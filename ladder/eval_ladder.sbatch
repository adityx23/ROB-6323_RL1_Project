#!/usr/bin/env bash
# Eval one ladder checkpoint on OCBench stacking: 200 fixed seeds (12345..12544, same as the overnight eval).
#   sbatch --array=0-8%2 eval_ladder.sbatch        (index -> step in ladder_steps.txt)
#SBATCH --job-name=ldr-eval
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
source $PROJ/overnight/env.sh
L=$PROJ/ladder
STEPS=($(cat $L/ladder_steps.txt)); STEP=${STEP:-${STEPS[$SLURM_ARRAY_TASK_ID]}}
CK=$RLFT_ROOT/runs/ladder/checkpoints/pi05_ocbs3k/run/$STEP
OUT=$RLFT_ROOT/runs/ladder/eval; mkdir -p $OUT/logs
POL=pi05_s$STEP; JID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
CT="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
echo "start=$(date -Is) host=$(hostname) job=$JID step=$STEP ckpt=$CK"
[ -d $CK/params ] || { echo "missing $CK/params"; exit 1; }
if grep -q "\"policy\": \"$POL\"" $OUT/results.jsonl 2>/dev/null; then echo "$POL already evaluated"; exit 0; fi
PORT=$((8100 + SLURM_JOB_ID % 800))
cd $L
$PY pi05_ladder.py serve $CK $PORT > $OUT/logs/server-$POL-$JID.log 2>&1 &
SP=$!
for i in $(seq 1 360); do $PY -c "import socket;socket.create_connection(('127.0.0.1',$PORT),1)" 2>/dev/null && break; sleep 5; done
echo "server up $(date -Is)"
CP=()
for k in 0 1 2 3; do
  (cd $RLFT_ROOT && $CT $PY $PROJ/overnight/ocb_eval.py --policy $POL --port $PORT \
    --episodes 50 --seed $((12345 + 50*k)) --replan-rows 5 --out $OUT) > $OUT/logs/client-$POL-$k-$JID.log 2>&1 &
  CP+=($!)
done
wait "${CP[@]}"; kill $SP 2>/dev/null
grep "\"policy\": \"$POL\"" $OUT/results.jsonl | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin]; print(f'$POL: {len(r)} episodes, success {sum(x[\"success\"] for x in r)}/{len(r)} = {sum(x[\"success\"] for x in r)/max(1,len(r)):.3f}')" | tee -a $OUT/RESULTS.txt
echo "end=$(date -Is)"
