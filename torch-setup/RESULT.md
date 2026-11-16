# Installation result — 2026-10-31

Completed successfully; runtime verification is limited to CPU imports.

## Job and evidence

- Job `21529922`, name `rlft-setup`, account `torch_pr_50_general` (user authorized all listed accounts for this workload; general account selected).
- Resources: one node, one task, two CPUs, 8 GB RAM, no GPUs, one-hour limit.
- Node `cs622.hpc.nyu.edu`; SLURM node `cs622` matched the short hostname.
- State `COMPLETED`; exit code `0:0`; elapsed `00:07:19`.
- `seff`: CPU time `00:00:59`, efficiency 6.72%; peak memory 8.00 GB, 99.96% of request. `sacct` batch MaxRSS `8385108K`. Memory was close to the limit; future setup resource proposals should account for this measurement. Download/filesystem wait dominated wall time; additional CPUs are not justified by this result.
- Output: `/scratch/aap9348/rlft-bench/logs/rlft-setup-21529922.out`.
- Error/install progress: `/scratch/aap9348/rlft-bench/logs/rlft-setup-21529922.err`. No error, traceback, or warning matches found in the logs.

## Environment and checks

- Native environment: `/scratch/aap9348/rlft-bench/.venv`, managed CPython 3.11.14.
- Bootstrap: uv 0.9.5 under `environments/uv-0.9.5`; Python distribution under `environments/python`.
- `uv sync --frozen --python 3.11` installed 220 platform-applicable packages.
- CPU imports passed for JAX 0.5.3, PyTorch 2.7.1+cu128, MuJoCo 3.17.0, mujoco-warp 3.17.0, OCBench 1.0.2, and OpenPI 0.1.0. JAX reported `[CpuDevice(id=0)]` with CPU backend explicitly selected.
- Lockfile SHA-256 stayed `dd75562b8cf751df99b5130de27dbad29ad9133d85ea4f7a11554e5383373f44`.
- Shell syntax checks for `setup.sbatch` and `run.sh` passed (exit zero).
- GPU simulation, EGL rendering, numerical/unit/full tests, training, and inference were not performed. No dataset or model weights were downloaded. Compatibility with a specific lab-PC OpenPI revision is unknown; the recorded default-branch revision passed imports.

## Files, storage, and commands

- Read setup script, relevant repository metadata/configuration, installation logs, Git status/revisions, and narrow scheduler/quota/module output. No secrets or protected research data were read.
- Home files added/updated: `setup.sbatch`, `run.sh`, `README.md`, this `RESULT.md` under `/home/aap9348/rlft-bench-torch`.
- Scratch changes: `.venv`, `cache/`, `environments/`, `logs/`, `tmp/` under the benchmark repository. The four latter directories appear as untracked generated files; `.venv` is already ignored. No tracked source or dependency pins changed; OpenPI checkout is clean. No pre-existing user changes were discarded.
- Commands executed: `hostname`, narrow SLURM marker reads, `my_slurm_accounts`, `myquota`, narrow file reads/listings, `git status`/`sha256sum`, `bash -n`, `mkdir -p` for logs, approved `sbatch`, restrained `squeue`/`sacct`/`seff`, and log `tail`/`rg` checks. The approved batch executed the recorded module activation, scratch directory creation, pip uv bootstrap, frozen uv sync, and Python import/version checks.
- Coordination/editing/checks ran on login node `torch-login-b-3`; package installation and imports ran inside the approved compute job on `cs622`.
- Quotas passed before installation and were rechecked afterward. Afterward: HOME 8 GB / 50 GB, 19,454 / 30,000 files; SCRATCH 2.95 TB / 5 TB, 1,185,137 / 5,000,000 files. Quota figures cover all user activity and are rounded; they are not an isolated measurement of this installation.
- Caches and temporary files were scoped to the batch/wrapper process; shell startup files and Codex SQLite configuration were not changed. Scratch content is not backed up; these small setup records are retained in backed-up home storage and repository source exists upstream.

## Next steps

Use `bash /home/aap9348/rlft-bench-torch/run.sh python ...` inside an approved compute allocation. The wrapper was syntax checked, but not executed as a workload in this job. Future allocations/submissions, GPU validation, dataset/model downloads, deletion, or environment updates require the approvals in the user's Torch policy. No automatic follow-up job is scheduled.
