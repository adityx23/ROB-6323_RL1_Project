#!/usr/bin/env bash
# CPU job: export usable diverse demos -> LeRobot local/ocb_diverse_1k, then compute its norm stats (pi05_diverse.py).
#SBATCH --job-name=dv-export
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
echo "start=$(date -Is) host=$(hostname)"
bash $PROJ/torch-setup/run.sh python $PROJ/diverse/export_diverse.py --num ${NUM:-1000}
source $PROJ/overnight/env.sh
export JAX_PLATFORMS=cpu
cd $PROJ/diverse && $PY pi05_diverse.py norm
find $RLFT_ROOT/runs/diverse/assets -name norm_stats.json
echo "end=$(date -Is)"
