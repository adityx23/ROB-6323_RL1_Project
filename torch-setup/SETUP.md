# Torch setup status

Installation completed successfully on 2026-10-31. Setup job `21529922` ran on CPU node `cs622`: SLURM `COMPLETED`, exit code `0:0`, elapsed `00:07:19`. See `RESULT.md` for verification and limits.

- Benchmark: `/scratch/aap9348/rlft-bench`, revision beginning `47b9571`.
- Dependency: `/scratch/aap9348/lair-openpi`, revision `82dd4279806eb994a48803b6152b9317c55f7c22`, branch `lair/main`.
- Account: `torch_pr_50_general`; user confirmed all listed accounts are authorized.
- Proposed installation: one node/task, two CPUs, 8 GB RAM, one hour, no GPU.
- Install scope: uv 0.9.5, managed Python 3.11, benchmark's frozen lockfile (including its pinned LeRobot Git dependency).
- Network sources: PyPI/Python Hosted, GitHub (Python distribution and LeRobot), PyTorch wheel hosting.
- Environment, caches, and temporary files stay inside the scratch benchmark directory. No shell startup edits.
- Estimate: 30–60 GB and potentially hundreds of thousands of files; not measured.
- CPU import checks passed. GPU simulation/rendering, tests, datasets, and weights were not run/downloaded.
- Submodules were not initialized; benchmark package installation uses the two editable OpenPI packages.

Approved submission command (already executed; do not resubmit without inspecting the existing environment):

```bash
mkdir -p /scratch/aap9348/rlft-bench/logs
cd /scratch/aap9348/rlft-bench
sbatch /home/aap9348/rlft-bench-torch/setup.sbatch
```

Logs: `/scratch/aap9348/rlft-bench/logs/rlft-setup-JOBID.out` and `.err`.

Use the companion wrapper inside a separately approved compute allocation:

```bash
bash /home/aap9348/rlft-bench-torch/run.sh python YOUR_SCRIPT.py
```

It runs from the benchmark root with `uv run --frozen --no-sync`, allocation-sized thread limits, EGL settings, and scratch cache/data/run paths. It refuses to run on a login node and does not submit jobs. Dataset and model downloads require separate approval. For CPU-only work, set `JAX_PLATFORMS=cpu` for the command.

These setup instructions are retained in backed-up home storage; source is also retained in the upstream Git repositories. Scratch is not backed up. The exact lab-PC OpenPI revision is unknown; compatibility is not established until runtime checks pass. The upstream Torch scripts still contain another user's paths and account: use neither them nor their launchers without adapting and reviewing them.
