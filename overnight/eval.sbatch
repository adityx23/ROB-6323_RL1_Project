#!/usr/bin/env bash
# Eval one policy on one benchmark. sbatch --export=ALL,POLICY=pi05|act,BENCH=l90|ocbs eval.sbatch
#SBATCH --job-name=ovn-eval
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
source $PROJ/overnight/env.sh
OUT=${OUT:-$RUNS/eval}; mkdir -p $OUT/logs
CT="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID policy=$POLICY bench=$BENCH"
cd $OVN
wait_port() { for i in $(seq 1 360); do $PY -c "import socket;socket.create_connection(('127.0.0.1',$1),1)" 2>/dev/null && return 0; sleep 5; done; return 1; }
SLOG=$OUT/logs/server-$POLICY-$BENCH-$SLURM_JOB_ID.log
if [ $POLICY = pi05 ]; then
  CK=$RUNS/checkpoints/pi05_$BENCH/run/$(ls $RUNS/checkpoints/pi05_$BENCH/run | grep -E "^[0-9]+$" | sort -n | tail -1)
  echo "checkpoint $CK"
  $PY ovn_pi05.py serve pi05_$BENCH $CK 8000 > $SLOG 2>&1 &
else
  echo "act run $(cat $RUNS/act_$BENCH/progress.json)"
  $PY act_serve.py $RUNS/act_$BENCH 8000 > $SLOG 2>&1 &
fi
SP=$!
wait_port 8000 || { echo "server did not come up"; tail -40 $SLOG; exit 1; }
echo "server up $(date -Is)"
CP=()
if [ $BENCH = l90 ]; then   # 4 LIBERO-90 stacking tasks x 50 episodes, one client per task
  for t in 16 17 63 64; do
    PYTHONPATH=$LIBERO_SRC $CT $LIBERO_PY libero_eval.py --policy $POLICY --suite libero_90 --suite-label l90_stack \
      --tasks $t --trials 50 --replan 5 --res 128 --out $OUT > $OUT/logs/client-$POLICY-l90-t$t-$SLURM_JOB_ID.log 2>&1 &
    CP+=($!)
  done
else                        # OCBench stacking: 200 seeded episodes, 4 clients x 50
  for k in 0 1 2 3; do
    (cd $RLFT_ROOT && $CT $PY $OVN/ocb_eval.py --policy $POLICY --port 8000 --episodes 50 \
      --seed $((12345 + 50*k)) --replan-rows 5 --out $OUT) > $OUT/logs/client-$POLICY-ocbs-$k-$SLURM_JOB_ID.log 2>&1 &
    CP+=($!)
  done
fi
wait "${CP[@]}"
kill $SP 2>/dev/null
grep "\"policy\": \"$POLICY\"" $OUT/results.jsonl | grep "\"suite\": \"$( [ $BENCH = l90 ] && echo l90_stack || echo ocb_stack )\"" | \
  python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin]; print(f'{len(r)} episodes, success {sum(x[\"success\"] for x in r)/max(1,len(r)):.3f}')"
echo "end=$(date -Is)"
