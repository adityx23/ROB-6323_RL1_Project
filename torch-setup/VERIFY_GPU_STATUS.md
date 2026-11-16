# GPU verification status — updated 2026-11-03

**Retry job `21746503` (`rlft-verify-gpu-retry`) submitted 2026-11-03, PENDING (QOS `gpu48`).**

First job `21532910` ran on `gl049` (NVIDIA L40S) and ended FAILED (exit 1:0, 00:18:43):
cuda ✅, simulation-render ✅, generate-data ✅, units ❌ (73 passed, 3 skipped, 1 failed);
base and RL workflows never ran.

The failure was `tests/test_actions.py::test_rel_abs_rel_roundtrip_on_dataset`: its conversion
assertions passed, but its `clipped < 5%` statistic failed (158/2424 rows; per-joint clipped rows
`[0 0 0 0 0 147 11]`, mostly wrist joint 5). The test's fixture targets the released dataset; we ran
it on 4 generated validation episodes. Treated as a data-dependent deviation, not an install fault;
unconfirmed against released data.

Retry (`verify_gpu_retry.sbatch` / `verify_gpu_retry.py`): same resources (1 GPU, 4 CPUs, 16 GB)
with 1:50 limit; copies data from `gpu-21532910/data`; runs cuda, units (that one test deselected),
base-workflow, rl-workflows (all 8 methods). Outputs: `/scratch/aap9348/rlft-bench/runs/verification/gpu-21746503/`,
logs `/scratch/aap9348/rlft-bench/logs/rlft-verify-gpu-retry-21746503.{out,err}`.

---

## Original status record (2026-10-31)


**Submitted, pending; full workflow remains unverified.**

- User approved the prepared single-GPU job, data generation, and tests.
- Job `21532910`, name `rlft-verify-gpu`, account `torch_pr_50_general` (user-authorized workload account).
- Requested one node/task, four CPUs, 16 GB host RAM, one generic GPU, 90 minutes; no partition or GPU type/QOS forced.
- Scheduler automatically selected QOS `gpu48`. Latest `squeue` reason: `QOSMaxGRESPerUser`. Earlier pending reasons included Resources. No node or GPU has been allocated.
- `sacct`: `PENDING`, elapsed zero, AllocCPUS zero. An exit-code field of `0:0` while pending does not establish success.
- `squeue --me` showed only this pending job, no running user jobs.
- At the diagnostic check, `show_slurm_qos` displayed `gpu48` MaxTRESPU `gres/gpu=0`; a direct `sacctmgr show qos` query displayed the general QOS limit `gres/gpu=16`. These sources differ; the exact effective user restriction and its cause are unverified. The scheduler's pending reason is the direct evidence preventing execution. No attempt was made to bypass or change the limit, switch accounts/QOS, cancel, or resubmit.
- The job remains queued and may start automatically if scheduler conditions change. No unattended monitoring loop or automatic resubmission was created.

## Outputs and verification

Expected scheduler logs (not created yet at inspection):

- `/scratch/aap9348/rlft-bench/logs/rlft-verify-gpu-21532910.out`
- `/scratch/aap9348/rlft-bench/logs/rlft-verify-gpu-21532910.err`

Expected stage outputs after execution: `/scratch/aap9348/rlft-bench/runs/verification/gpu-21532910/`.

No GPU checks, OCBench data generation, unit/smoke tests, rendering, GPU utilization measurements, or full-workflow artifacts have been produced by this job. Prepared shell syntax passed; runtime is not tested. Earlier CPU verification remains valid: 35 tests passed, five unavailable-reference/data skips, toy base/RL/evaluation/checkpoint/resume passed.

## Activity and preservation

Coordination and narrow checks occurred on login node `torch-login-b-3`; no compute workload ran on it. Read account/quota/node status, reviewed the prepared batch script, queried scheduler/QOS/account associations and source status. Executed `hostname`, narrow SLURM marker reads, `my_slurm_accounts`, `myquota`, `bash -n`, `git status`/`git diff --stat`, approved `sbatch`, restrained `squeue`/`sacct`, `show_slurm_qos`, read-only `sacctmgr`, and guarded log listings/reads. Initial missing-log checks reported files absent because the job had not started.

Home files changed: this status report and `VERIFY_GPU_PLAN.md`. No repository tracked source or dependency pins changed; pre-existing generated directories and user work were preserved. No environment/package changes, downloads, deletion, cancellation, or additional scheduler submissions.

Pre-submission quota passed: HOME 8 GB / 50 GB and 19,463 / 30,000 files; SCRATCH 2.95 TB / 5 TB and 1,189,439 / 5,000,000 files. No material data writes occurred afterward.

## Remaining work

GPU verification depends on scheduler eligibility/resources becoming available. Once the queued job can run, inspect its state, per-stage logs/artifacts, exit code, and efficiency. If any stage fails, diagnose before claiming success or proposing another job. Any changed allocation/submission or cancellation still requires approval under the user's Torch policy. No additional approval is needed merely for the already approved queued job to start.
