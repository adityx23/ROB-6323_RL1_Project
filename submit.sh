#!/usr/bin/env bash
# Submit a job script of this repo the way all of our jobs were run: from RLFT_ROOT (relative log paths
# logs/%x-%j.{out,err}), with PROJ/RLFT_ROOT/LAIR_OPENPI exported into the job.
#   ./submit.sh [sbatch options] <script.sbatch|.slurm> [script args]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
args=()
for a in "$@"; do
  if [[ "$a" == *.sbatch || "$a" == *.slurm ]] && [ -f "$a" ]; then a=$(cd "$(dirname "$a")" && pwd)/$(basename "$a"); fi
  args+=("$a")
done
mkdir -p "$RLFT_ROOT/logs"
cd "$RLFT_ROOT"
exec sbatch --export=ALL ${ACCOUNT:+--account=$ACCOUNT} "${args[@]}"
