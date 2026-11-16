# Proposed single-GPU workflow validation

Approved and submitted as job `21532910` on 2026-10-31. Latest inspected state: `PENDING`, reason `QOSMaxGRESPerUser`, automatically assigned QOS `gpu48`. No compute node/GPU allocated; no stage has executed. The CPU prerequisite passed across jobs 21531365 and 21531659. See `VERIFY_GPU_STATUS.md`.

Account: `torch_pr_50_general`, explicitly authorized for this workload. Account association, current `show_slurm_qos`, and GPU inventory were inspected. The association lists normal QOS with no displayed TRES limits. GPU access under this account has not yet been demonstrated; no GPU model, partition, or explicit QOS is assumed/requested.

Resource proposal: one node/task, four CPUs, 16 GB host RAM, one GPU (`--gres=gpu:1`), 90 minutes. Four CPUs allow simulation, rendering, and compilation with bounded threading; 16 GB covers the small generated visual dataset, replay/cache, and test subprocesses. CPU validation itself peaked below 2 GB. The upstream wiki estimates about 40 minutes for all-method smoke tests; the additional time covers first-use GPU kernel compilation, simulation checks, and data generation. No multiple GPUs/nodes or automatic resubmission.

Stages, sequentially stopping on failure:

1. Confirm JAX and PyTorch see and compute on CUDA; log GPU name/driver/memory.
2. Run the upstream simulation/render spike with 8 state worlds (reduced from 256) and its 8 visual worlds; assert finite JAX math, CUDA torch operation, simulation throughput, EGL rendering, and nonblank images.
3. Generate synthetic OCBench oracle datasets locally: 16 state training + 4 validation episodes and 8 visual training + 4 validation episodes, seed 0, image interval 5, two parallel worlds. Stop if any split has no successful examples. Estimated generated/cached/test artifacts: under 10 GB, potentially several thousand files, all scratch.
4. Run all unit tests, including GPU/environment/critic/data tests.
5. Run upstream base-policy state/visual smoke tests, checkpoint resume, videos, and probe construction. Exclude the deliberate preemption/requeue test.
6. Run all eight RL methods in state and visual modes, including their upstream evaluation/video/diagnostic/checkpoint/resume assertions. The upstream configs request up to 100 state / 30 visual demonstrations but the loader caps these to the available successful generated episodes; this is intentionally reduced-data execution validation.

No datasets, model weights, or new packages will be downloaded. Hub offline flags are set. Upstream preemption-driven `scontrol requeue` is disabled by withholding `SLURM_JOB_ID` from test/training subprocesses; the monitoring parent retains and logs its allocation identity. GPU usage/memory samples are logged every 30 seconds during longer stages. No changes to upstream source, dependency pins, launchers, or user startup files.

Outputs: `/scratch/aap9348/rlft-bench/runs/verification/gpu-JOBID/` (per-stage logs, GPU samples, generated data, tests, artifacts, and final `result.json` only on success). Scheduler logs: `/scratch/aap9348/rlft-bench/logs/rlft-verify-gpu-JOBID.out` and `.err`.

Files prepared in backed-up home: `verify_gpu.sbatch`, `verify_gpu.py`, `generate_verification_data.py`, and this plan. Shell/Python syntax checks passed; runtime behavior is not tested. GPU allocation, data generation, and test execution require approval before submitting the prepared batch job. The full scientific benchmark, released-data download path, performance claims, preemption behavior, dashboards, and multi-GPU scaling are outside this reduced workflow verification.
