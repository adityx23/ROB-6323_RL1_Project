#!/usr/bin/env bash
# Queue the data download, then three base policies that start only if the download succeeds.
# Targets from experiments.yaml E4 (chunk-frame abs ladder): ~15% -> 125 demos, ~35% -> 330 demos
# (100k steps each); ~55% -> 1,000 demos at 200k steps.
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd $RLFT_ROOT
mkdir -p logs
here=$PROJ
dl=$(sbatch --parsable "$here/download_data.sbatch")
echo "download: $dl"
for spec in 125:100000 330:100000 1000:200000; do
  n=${spec%:*}; steps=${spec#*:}
  id=$(sbatch --parsable --dependency=afterok:"$dl" --job-name="rlft-base-n$n" \
       --export=ALL,NDEMOS="$n",STEPS="$steps" "$here/train_base.sbatch")
  echo "base n=$n steps=$steps: $id"
done
squeue --me
