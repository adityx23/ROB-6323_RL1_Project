#!/usr/bin/env bash
# Build the rollout gallery (index.html + small mp4s) and a text summary once all evals have ended.
#SBATCH --job-name=ovn-gallery
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -uo pipefail
source /home/aap9348/rlft-bench-torch/overnight/env.sh
cd $OVN
$PY build_gallery.py $RUNS/eval $RUNS/gallery
$PY - <<'E' | tee $RUNS/RESULTS.txt
import json, collections
c = collections.defaultdict(lambda: [0, 0])
for l in open("/scratch/aap9348/rlft-bench/runs/overnight/eval/results.jsonl"):
    r = json.loads(l); c[(r["policy"], r["suite"])][0] += r["success"]; c[(r["policy"], r["suite"])][1] += 1
for (p, s), (k, n) in sorted(c.items()):
    print(f"{p:6s} {s:10s} {k:4d}/{n:<4d} = {k / n:.3f}")
E
du -sh $RUNS/gallery
