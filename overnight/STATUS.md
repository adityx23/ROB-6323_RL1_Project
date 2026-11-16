# Overnight run (2026-11-08 → 2026-11-09): OCBench cube stacking, shared recipe

Task: OCBench block-double-task2 (stack_anywhere). Demos were regenerated at image interval 5 (4 parts, seeds 1-4),
and 200 successful ones were exported to `local/ocb_stack`: front + wrist cameras at 128 px, 10 Hz, absolute joint
targets.

Shared recipe (pi0.5 and ACT): 200 demos; 2 views at 128 px; chunk 10, execute 5; batch 32; 15k steps; AdamW peak
5e-5, 500 warmup, cosine to 5e-6; grad clip 1.0; no EMA. Arm actions are deltas from the chunk-start joints, gripper
absolute (both models). ACT backbone lr = lr/5. pi0.5: `pi05_base` weights, 2-GPU FSDP.

Eval: 200 seeded episodes (seeds 12345-12544). Each policy row is an absolute joint target held for 5 control steps;
the policy replans every 5 rows.

Scripts: `regen_ocb.sbatch` + `export_ocb_lerobot.py`, `train_pi05.sbatch` (`ovn_pi05.py`), `train_act.sbatch`
(`act_train.py`), `eval.sbatch` (`ocb_eval.py`, `act_serve.py`), `build_gallery.py`. Job ids: `jobs.txt`.

## Results (`RESULTS.txt`)

| Policy | OCBench stacking |
|---|---|
| pi0.5 | 66% (132/200) |
| ACT | 0% (0/200) |

## Notes

- The first OCBench regen (4x100 worlds on one GPU) died silently. Splitting it into 2 GPUs x 110 worlds fixed it.
- The policy servers dropped clients on websocket keepalive. Fix: pings disabled; clients reconnect and retry.
- Both pi0.5 runs first crashed at startup: openpi's `action_sequence_keys` default is "actions", our column is
  "action". Fixed in `ovn_pi05.py`.
- ACT's 0% is not a pipeline bug (`ocb_debug.py`, `ocb_replay.py`):
  - Offline ACT error is ~0.01 rad, versus 0.07 for a hold-still baseline, and eval proprio stays within the
    training ranges.
  - In closed loop the arm moves at about half the demo speed, and the gripper never closes in 600 steps.
  - Replaying the demo targets from the recorded start states stacks 6/6 per-step, and 5/6 with the eval's 5-step
    hold.
  - So ACT under this recipe (200 demos, chunk 10, 15k steps) fails on this long-horizon task (~1400 control
    steps).
