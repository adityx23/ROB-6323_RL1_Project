# AGENTS.md

Exact setup and run steps for this repo on the NYU Torch cluster, in the order we ran them. Everything below was run
through Slurm. Nothing heavy runs on a login node.

## 0. Layout and what is pinned

```
ROB-6323_RL1_Project/          this repo (job scripts, configs, notes); clone it on /scratch, not $HOME
  config.sh                    PROJ, RLFT_ROOT, LAIR_OPENPI, pinned commits, ACCOUNT  (source it first)
  submit.sh                    submit any job script the way we did (from RLFT_ROOT, env exported)
  torch-setup/  overnight/  ladder/
  external/rlft-bench          adityx23/rlft-bench  @ ee7968d (branch binary-advc)   = RLFT_ROOT
  external/lair-openpi         adityx23/lair-openpi @ 82dd427 (branch lair/main)     = LAIR_OPENPI
```

* rlft-bench's `pyproject.toml` installs `../lair-openpi` (editable), so the two checkouts must be siblings.
* Data, caches, venvs, runs and logs all live **inside `RLFT_ROOT`** (`data/`, `cache/`, `environments/`, `.venv/`,
  `runs/`, `logs/`, `tmp/`). The full pipeline needs about 300 GB and a few hundred thousand files.
* Pinned software, all checked by `torch-setup/check_env.sh`:
  * **rlft-bench venv:** uv 0.9.5, CPython 3.11.14 (uv-managed), `uv sync --frozen` from rlft-bench's `uv.lock`.
    Key packages: jax 0.5.3, torch 2.7.1+cu128, mujoco 3.17.0, mujoco-warp 3.17.0, numpy<2. Full list:
    `torch-setup/rlft_venv_freeze.txt`.
  * **Weights:** `gs://openpi-assets/checkpoints/pi05_base/params` (12 GB) and the PaliGemma tokenizer, both
    stored in `$RLFT_ROOT/cache/openpi`.
* Slurm accounts: `torch_pr_50_tandon_priority` for jobs and `torch_pr_50_general` for setup. To use a different
  account, `export ACCOUNT=<yours>` before calling `submit.sh`.

## 1. Clone

```bash
cd /scratch/$USER
git clone https://github.com/adityx23/ROB-6323_RL1_Project.git
cd ROB-6323_RL1_Project
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init external/rlft-bench external/lair-openpi
source config.sh
```
Do not run `git submodule update --init --recursive`: lair-openpi's own submodules (`third_party/aloha`,
`third_party/libero`) are not used here. We only use OCBench.

## 2. Environments (CPU jobs)

```bash
source config.sh
./submit.sh torch-setup/setup.sbatch            # uv 0.9.5 bootstrap -> Python 3.11 -> uv sync --frozen (~10 min)
./submit.sh torch-setup/prefetch_weights.sbatch # pi05_base params + PaliGemma tokenizer -> cache/openpi
bash torch-setup/check_env.sh                   # must print IDENTICAL and the expected commits
```
* `setup.sbatch` refuses to run if `.venv` already exists, if the checkouts are at different commits, or if
  lair-openpi is not the sibling `../lair-openpi`. It never edits `uv.lock`.
* Never run an unfrozen `uv sync`, `uv lock` or `pip install` in the rlft-bench venv: jax, flax, numpy<2 and
  pyarrow have to match lair-openpi.
* All later Python runs inside jobs go through `torch-setup/run.sh` (for rlft-bench code) or `$PY` from
  `overnight/env.sh`. Both use `$RLFT_ROOT/.venv`.

## 3. OCBench cube stacking data (block-double-task2)

```bash
# 200-demo set first (parts p000-p003, seeds 1-4; 110 episodes per part, ~85% success), no export
./submit.sh --export=ALL,PARTS='0 1',NOEXPORT=1 overnight/regen_ocb.sbatch
./submit.sh --export=ALL,PARTS='2 3',NOEXPORT=1 overnight/regen_ocb.sbatch
# 3k set: parts p004-p033 (seeds 5-34), 2 parts per array task; skips parts that already exist
./submit.sh --array=0-14%2 --constraint=l40s ladder/regen_3k.sbatch
# LeRobot export: 3000 successful demos -> cache/huggingface/lerobot/local/ocb_stack_3k (CPU, ~6 h)
./submit.sh ladder/export_3k.sbatch
```
Output: `data/ocbench/regen/visual-block-double-task2-v0/i5/train-p000..p033.npz`. That's 3,740 episodes,
3,256 of them successful, 67 GB. The `ocb_stack_3k` export is 3,000 episodes, 835,556 frames at 10 fps, 15 GB.
* Run the regen on **L40S** (`--constraint=l40s`): about 13 min per task, versus about 40 min on H100. Node `gh003`
  crashed in EGL rendering (`Fatal Python error: Aborted`). Torch rejects `--exclude`; report bad nodes to
  hpc@nyu.edu.
* `export_3k.sbatch` runs on a CPU node. `export_3k.py` switches MuJoCo to `glfw` because CPU nodes have no EGL.
  It first checks itself bit-for-bit against the 200-demo `ocb_stack` export when that export exists.
* The 200-demo `local/ocb_stack` export (overnight run) is `overnight/export_ocb.sbatch`. It needs a GPU node.

## 4. π0.5 checkpoint ladder (ladder/)

Config: `ladder/pi05_ladder.py`, built on `overnight/ovn_pi05.py`. That file starts from openpi's `pi05_libero`
TrainConfig only for the `pi05_base` weights and defaults, and replaces its data config with OCBench's.
* Data and model: `local/ocb_stack_3k`, π0.5 from `pi05_base`, action chunk 30, train-time RTC (delay
  ~ U[0, 15)), continuous state.
* Optimization: batch 128, 30k steps, AdamW, lr 5e-5 cosine to 5e-6 (500 warmup), grad clip 1.0, EMA 0.999.
* Compute: 4-GPU FSDP.
* Checkpoints: saved every 2500 steps. Kept: 5k, 10k, 15k, 20k, 25k, 29999 (`ladder/ladder_steps.txt`). For
  each kept step, `prune_ckpts.sh` deletes the optimizer state and keeps params + assets, except for the newest
  checkpoint.

```bash
J=$(./submit.sh --parsable ladder/norm_ladder.sbatch)                                   # norm stats, CPU
S=$(./submit.sh --parsable --dependency=afterok:$J ladder/smoke_ladder.sbatch)          # 60 steps, 4 GPUs
T=$(./submit.sh --parsable --dependency=afterok:$S ladder/train_ladder.sbatch)          # full run, resumable
./submit.sh --dependency=afterok:$T ladder/upload.sbatch   # public HF upload, needs ~/.config/ldr-hf/token (mode 600)
```
* Outputs: `runs/ladder/checkpoints/pi05_ocbs3k/run/<step>/` and `runs/ladder/assets/` (norm stats). The
  checkpoints are uploaded to https://huggingface.co/adityx23/pi05-ocb-stack-ladder.
* Never hold a GPU for CPU-only work (norm stats, export, downloads). Root has cancelled such jobs after about
  2 h.
* To resubmit `train_ladder.sbatch`, just submit it again: it resumes from the newest checkpoint.
* `eval_ladder.sbatch` and `eval_watcher.slurm` were written for the 10-step, non-RTC policy (the overnight eval
  protocol: replan every 5 rows, each row held 5 control steps, 200 seeds 12345..12544). They are **not yet
  adapted** to the H=30 RTC ladder.

## 5. Overnight run (overnight/, 200 OCBench demos; see overnight/STATUS.md)

```bash
./submit.sh overnight/export_ocb.sbatch                                                  # local/ocb_stack (GPU)
./submit.sh --export=ALL,NAME=pi05_ocbs overnight/train_pi05.sbatch
./submit.sh --export=ALL,POLICY=pi05,BENCH=ocbs overnight/eval.sbatch                    # or POLICY=act
./submit.sh --export=ALL,BENCH=ocbs overnight/train_act.sbatch
./submit.sh overnight/gallery.sbatch
```
Results (`overnight/RESULTS.txt`):

| Policy | OCBench stacking |
|---|---|
| π0.5 | 66% |
| ACT | 0% |

## 6. rlft-bench RL (torch-setup/)

`download_data.sbatch`, `train_base.sbatch`, `train_rl.sbatch`, `submit_bases.sh` and `test_binary_advc.sbatch`
run rlft-bench's own `scripts/` through `torch-setup/run.sh`. Submit them with `./submit.sh` from the repo. For
`submit_bases.sh`, run `source config.sh && cd $RLFT_ROOT && bash $PROJ/torch-setup/submit_bases.sh`.
rlft-bench's own rules apply: `external/rlft-bench/CLAUDE.md` and `docs/wiki/`.

## 7. Using the original checkouts

Our runs used `RLFT_ROOT=/scratch/aap9348/rlft-bench` and `LAIR_OPENPI=/scratch/aap9348/lair-openpi`, at the same
commits as the submodules. To reuse them: `export RLFT_ROOT=/scratch/aap9348/rlft-bench; source config.sh`.
The `*.md` status notes in `torch-setup/` and `overnight/` are historical records and keep those absolute paths.
