# Training status — 2026-11-03

Setting: block-single-task1, visual obs, chunk-frame abs actions, H = C = 25 (experiments.yaml E2/E4).
Account `torch_pr_50_general`. RL methods still to decide (user has new methods to add).

| Job | What | Depends on |
|---|---|---|
| 21747776 | download released shard 000 (visual + state) -> /scratch/aap9348/rlft-bench/data/ocbench | — |
| 21747777 | base n=125, 100k steps (~15% target) | download ok |
| 21747778 | base n=330, 100k steps (~35% target) | download ok |
| 21747779 | base n=1000, 200k steps (~55% target) | download ok |

Base runs: /scratch/aap9348/rlft-bench/runs/base/base_block-single-task1_visual_abs-chunk_H25_n<N>_<K>k/s0
Base jobs: 1 L40S, 8 CPUs, 64 GB, 12h; USR1 at -10 min -> checkpoint + self-requeue + resume.
Logs: /scratch/aap9348/rlft-bench/logs/rlft-download-<id>.out, rlft-base-n<N>-<id>.out

RL later: sbatch --export=ALL,BASE=runs/base/<name>/s0,METHOD=<m>[,SEED=0][,EXTRA="k=v"] train_rl.sbatch
(submit from /scratch/aap9348/rlft-bench). Untested until a base exists.
