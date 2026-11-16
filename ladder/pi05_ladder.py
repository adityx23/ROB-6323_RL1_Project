"""pi0.5 checkpoint ladder on OCBench cube stacking (block-double-task2), 3000 demos (local/ocb_stack_3k).

Transforms as the overnight pi05_ocbs run (overnight/ovn_pi05.py: front -> base view, wrist -> left wrist, right wrist
masked; arm actions as deltas from the chunk-start joints, gripper absolute). Changed per user (2026-11-16):
action chunk 30, train-time RTC (delay ~ U[0, 15)), batch 128, EMA 0.999, 30k steps, 4-GPU FSDP.
Checkpoints: saved every 2500 steps; orbax keeps multiples of 5000 (keep_period) plus the newest; prune_ckpts.sh
drops train_state from the ladder steps (ladder_steps.txt: 5k 10k 15k 20k 25k 29999) except the newest.
With EMA on, the saved `params` are the EMA weights.

    python pi05_ladder.py norm                    # compute + cache norm stats (CPU job)
    python pi05_ladder.py train [--resume]        # LDR_EXP (default run), LDR_STEPS (default 30000)
    python pi05_ladder.py serve <ckpt_dir> [port]
    python pi05_ladder.py infer <ckpt_dir>        # load a checkpoint, run one inference with and without RTC prefix
"""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, "/home/aap9348/rlft-bench-torch/overnight")

import ovn_pi05 as ovn  # noqa: E402  (puts lair-openpi on sys.path, disables websocket keepalive)
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as C  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402

NAME = "pi05_ocbs3k"
RUNS = "/scratch/aap9348/rlft-bench/runs/ladder"
STEPS = int(os.environ.get("LDR_STEPS", 30000))
H = 30


def get() -> C.TrainConfig:
    cfg = ovn.get("pi05_ocbs")
    return dataclasses.replace(
        cfg, name=NAME, exp_name=os.environ.get("LDR_EXP", "run"),
        data=dataclasses.replace(cfg.data, repo_id="local/ocb_stack_3k"),
        model=pi0_config.Pi0Config(pi05=True, action_horizon=H, discrete_state_input=False,
                                   rtc_train_enabled=True, rtc_train_min_delay=0, rtc_train_max_delay=15),
        batch_size=128, num_train_steps=STEPS, ema_decay=0.999,
        fsdp_devices=int(os.environ.get("LDR_FSDP", 4)), num_workers=int(os.environ.get("LDR_WORKERS", 16)),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=STEPS, decay_lr=5e-6),
        save_interval=int(os.environ.get("LDR_SAVE", 2500)), keep_period=int(os.environ.get("LDR_KEEP", 5000)),
        log_interval=50, wandb_enabled=False,
        checkpoint_base_dir=f"{RUNS}/checkpoints", assets_base_dir=f"{RUNS}/assets")


def _policy(cfg, ckpt):
    from openpi.policies import policy_config

    policy = policy_config.create_trained_policy(cfg, ckpt)
    _infer = policy.infer
    policy.infer = lambda obs, **kw: _infer({k: v for k, v in obs.items() if not k.endswith("_raw")}, **kw)
    return policy


if __name__ == "__main__":
    mode = sys.argv[1]
    cfg = get()
    if mode == "norm":
        import openpi.training.norm_stats as _ns

        _ns.ensure(cfg)
    elif mode == "train":
        import train as _train

        resume = "--resume" in sys.argv
        _train.main(dataclasses.replace(cfg, resume=resume, overwrite=not resume))
    elif mode == "serve":
        from openpi.serving import websocket_policy_server

        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        websocket_policy_server.WebsocketPolicyServer(_policy(cfg, sys.argv[2]), host="0.0.0.0", port=port).serve_forever()
    elif mode == "infer":
        import inspect

        import numpy as np

        policy = _policy(cfg, sys.argv[2])
        rng = np.random.default_rng(0)
        obs = {"observation/front": rng.integers(0, 255, (128, 128, 3), np.uint8),
               "observation/wrist": rng.integers(0, 255, (128, 128, 3), np.uint8),
               "observation/state": np.zeros(19, np.float32), "prompt": "stack the two cubes"}
        a = np.asarray(policy.infer(obs)["actions"])
        print("infer OK, actions", a.shape, "finite", bool(np.isfinite(a).all()), flush=True)
        print("Policy.infer signature:", inspect.signature(type(policy).infer if hasattr(type(policy), "infer") else policy.infer))
