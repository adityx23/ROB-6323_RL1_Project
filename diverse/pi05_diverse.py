"""pi0.5 on the diverse colored cube-stacking set (local/ocb_diverse_1k), full fine-tune on ONE H200 (141 GB).

Same recipe as the 3k ladder (ladder/pi05_ladder.py): pi05_base weights, action chunk 30, train-time RTC
(delay ~ U[0, 15)), continuous state, AdamW 5e-5 cosine to 5e-6, grad clip 1.0, EMA 0.999, same transforms. The prompt
comes from each episode's task string ("stack the red cube on top of the blue cube"). Differences: 1 GPU (no FSDP),
batch DV_BATCH (default 64), DV_STEPS (default 15000, ~3.5 passes over ~1k demos at batch 64), checkpoints at
5k/10k/15k. Full fine-tuning does not fit on one 80 GB GPU (OOM on H100, Nov 20), so this needs an H200.

    python pi05_diverse.py norm | train [--resume] | serve <ckpt> [port] | infer <ckpt>
"""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.environ["PROJ"], "ladder"))

import pi05_ladder as ladder  # noqa: E402  (sets up openpi imports exactly like the ladder)
import openpi.training.optimizer as _optimizer  # noqa: E402

NAME = "pi05_ocbdiv1k"
RUNS = os.environ["RLFT_ROOT"] + "/runs/diverse"
STEPS = int(os.environ.get("DV_STEPS", 15000))


def get():
    cfg = ladder.get()
    return dataclasses.replace(
        cfg, name=NAME, exp_name=os.environ.get("DV_EXP", "run"),
        data=dataclasses.replace(cfg.data, repo_id=os.environ.get("DV_REPO", "local/ocb_diverse_1k")),
        batch_size=int(os.environ.get("DV_BATCH", 64)), num_train_steps=STEPS,
        fsdp_devices=1, num_workers=int(os.environ.get("DV_WORKERS", 16)),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=min(500, STEPS // 10), peak_lr=5e-5,
                                                   decay_steps=STEPS, decay_lr=5e-6),
        save_interval=int(os.environ.get("DV_SAVE", 2500)), keep_period=int(os.environ.get("DV_KEEP", 5000)),
        checkpoint_base_dir=f"{RUNS}/checkpoints", assets_base_dir=f"{RUNS}/assets")


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
        websocket_policy_server.WebsocketPolicyServer(ladder._policy(cfg, sys.argv[2]), host="0.0.0.0",
                                                      port=port).serve_forever()
    elif mode == "infer":
        import numpy as np

        policy = ladder._policy(cfg, sys.argv[2])
        rng = np.random.default_rng(0)
        obs = {"observation/front": rng.integers(0, 255, (128, 128, 3), np.uint8),
               "observation/wrist": rng.integers(0, 255, (128, 128, 3), np.uint8),
               "observation/state": np.zeros(19, np.float32),
               "prompt": "stack the red cube on top of the blue cube"}
        a = np.asarray(policy.infer(obs)["actions"])
        print("infer OK, actions", a.shape, "finite", bool(np.isfinite(a).all()), flush=True)
