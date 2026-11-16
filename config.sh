# Shared paths for every script in this repo.  Usage:  source config.sh
# Default layout = a recursive clone of this repo: external/rlft-bench and external/lair-openpi side by side
# (rlft-bench's pyproject expects ../lair-openpi). Data, caches, venvs, runs and logs all live inside RLFT_ROOT,
# so put the clone on /scratch, not $HOME. Override RLFT_ROOT (and LAIR_OPENPI) to use other checkouts.
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJ
export RLFT_ROOT=${RLFT_ROOT:-$PROJ/external/rlft-bench}
export LAIR_OPENPI=${LAIR_OPENPI:-$(dirname "$RLFT_ROOT")/lair-openpi}
# Commits everything was run at (also pinned as the submodule commits).
export RLFT_COMMIT=ee7968d795e0d0d8d2c827cfbd32892c1e0a4a63    # adityx23/rlft-bench, branch binary-advc
export LAIR_COMMIT=82dd4279806eb994a48803b6152b9317c55f7c22    # adityx23/lair-openpi, branch lair/main
# Slurm account: the scripts default to torch_pr_50_tandon_priority (setup: torch_pr_50_general);
# set ACCOUNT to override it for every job submitted with submit.sh.
export ACCOUNT=${ACCOUNT:-}
