# Overnight run (2026-11-08 → 2026-11-09 10:00) — cube/bowl STACKING, shared recipe
Pivot at 23:00: first plan (LIBERO 4 suites + OCBench block-single) cancelled per user.

Tasks
- LIBERO-90 bowl stacking, tasks 16, 17, 63, 64 (50 demos each, HDF5 → local/libero90_stack, 128px, 20 Hz, no-ops dropped, no image flip)
- OCBench block-double-task2 (stack_anywhere): regen at image interval 5 → 200 successful demos → local/ocb_stack (front+wrist 128px, 10 Hz, abs joint targets)

Shared recipe (all 4 runs): 200 demos/benchmark; 2 views 128px; chunk 10 / execute 5; batch 32; 15k steps;
AdamW peak 5e-5, 500 warmup, cosine → 5e-6; grad clip 1.0; no EMA. OCBench arm actions = delta from chunk-start
joints (both models). ACT: task one-hot (LIBERO-90 4 tasks); backbone lr = lr/5. pi0.5: pi05_base, 2-GPU FSDP.
Eval: LIBERO-90 50 eps/task (fixed init states) = 200; OCBench 200 seeds. Gallery: build_gallery.py.

Scripts: convert_libero90.py, regen_ocb.sbatch + export_ocb_lerobot.py, train_pi05.sbatch (ovn_pi05.py), train_act.sbatch (act_train.py),
eval.sbatch (libero_eval.py, ocb_eval.py, act_serve.py). Job ids: jobs.txt. Logs: /scratch/aap9348/rlft-bench/logs/ovn-*.

## Unattended chain (all in SLURM; nothing depends on the Claude session)
regen_a + regen_b (OCBench stacking demos) → export_ocb → act_ocbs, pi05_ocbs (training)
conv_l90 (done) → act_l90 (done, 15k steps) + pi05_l90 (training)
Each training job → its eval (ACT: afterok; pi0.5: afterany). watchdog at 07:00 cancels any pi0.5 training still
running so its eval starts on the latest checkpoint (saved every 1000 steps).
All 4 evals → gallery job → /scratch/aap9348/rlft-bench/runs/overnight/gallery/index.html (+ v/*.mp4)
and /scratch/aap9348/rlft-bench/runs/overnight/RESULTS.txt (success rate per policy × benchmark).
Job ids: jobs.txt.

## Check / recover
squeue -u aap9348                         # what's queued/running
cat /scratch/aap9348/rlft-bench/runs/overnight/RESULTS.txt
tail /scratch/aap9348/rlft-bench/logs/ovn-<name>-<jobid>.{out,err}
Re-run one eval:  cd ~/rlft-bench-torch/overnight && sbatch --export=ALL,POLICY=pi05|act,BENCH=l90|ocbs eval.sbatch
Rebuild gallery:  sbatch gallery.sbatch
View gallery: copy the gallery folder to your laptop (scp -r torch:/scratch/aap9348/rlft-bench/runs/overnight/gallery .) and open index.html,
or ask Claude to publish it as an artifact.

## Fixes made during the night
- LIBERO eval clients ran ~100x slow from thread oversubscription → single-threaded math libs in the clients.
- Policy servers dropped clients on websocket keepalive → pings disabled, clients reconnect + retry.
- First OCBench regen (4x100 worlds on one GPU) died silently → split to 2 GPUs x 110 worlds.

## 2026-11-09 morning
Overnight result: ACT l90 71.6% (144/200); ACT ocbs 0/200 (all episodes timed out, suspected bug); both pi0.5 runs
crashed at startup (openpi action_sequence_keys default "actions" vs our "action" column) → fixed in ovn_pi05.py,
pipeline validated on real samples. Published ACT gallery: https://claude.ai/artifact/3xB81hu22Jgbeb2ryBhwaG
Retry chain (11:40): pi05_l90_v2, pi05_ocbs_v2 (6.5 h limit) → evals (afterany) → gallery_v2 (runs/overnight/gallery).
ACT OCBench debug: ocb_debug.sbatch → runs/overnight/ocb_debug/report.txt + pngs. Checked and ruled out: the 5-step
hold of per-step targets (row targets already lead q by more than the 5-step motion).
Note: pi05_ocbs trains on the same OCBench data as ACT; if the debug finds a data bug, it must be retrained.
ACT OCBench debug result (11:45): NOT a pipeline bug. (A) offline ACT error ~0.01 rad vs 0.07 for a hold-still
baseline; (B) eval proprio within training ranges; (C) closed loop: arm moves ~half demo speed, gripper never closes in
600 steps; replay test (ocb_replay.py): demo targets replayed from recorded start states stack 6/6 per-step and 5/6 with
the eval's 5-step hold. → ACT under the shared recipe (200 demos, chunk 10, 15k steps) fails on this long-horizon
(~1400 control steps) task. pi05_ocbs uses the same data/execution path and is valid.
