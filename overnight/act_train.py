"""ACT (LeRobot 0.1.0 ACTPolicy) trainer, shared overnight recipe.

    python act_train.py --bench ocbs --out RUN_DIR [--max-hours H]

Shared recipe (same as the pi0.5 runs): batch 32, 15k steps, AdamW, peak lr 5e-5 with 500 warmup steps and
cosine decay to 5e-6, grad clip 1.0, no EMA, action chunk 10 (executed 5 at eval). ACT's ResNet backbone
uses 1/5 of the lr (ACT's usual backbone split), on the same schedule.

ocbs : local/ocb_stack (OCBench block-double-task2). Arm actions are trained as deltas from the chunk-start
       joint positions (state[:6]), the same transform pi0.5 uses; gripper stays absolute.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

BENCH = {
    "ocbs": dict(repo="local/ocb_stack", images=["observation.images.front", "observation.images.wrist"],
                 one_hot=False, arm_delta=True),
}


def lr_lambda(step, warmup=500, total=15000, peak=5e-5, end=5e-6):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return (end + 0.5 * (peak - end) * (1 + math.cos(math.pi * p))) / peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCH), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--max-hours", type=float, default=7.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=1000)
    a = ap.parse_args()
    t_start = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = BENCH[a.bench]
    meta = LeRobotDatasetMetadata(B["repo"])
    fps = meta.fps
    ds = LeRobotDataset(B["repo"], delta_timestamps={"action": [i / fps for i in range(a.chunk)]})
    print(f"{B['repo']}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {fps}", flush=True)

    st = meta.stats
    tens = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    in_feats, stats = {}, {}
    for k in B["images"]:
        h, w, _ = meta.features[k]["shape"]
        in_feats[k] = PolicyFeature(FeatureType.VISUAL, (3, h, w))
        stats[k] = {s: tens(st[k][s]) for s in ("mean", "std")}
    sdim = meta.features["observation.state"]["shape"][0]
    in_feats["observation.state"] = PolicyFeature(FeatureType.STATE, (sdim,))
    stats["observation.state"] = {s: tens(st["observation.state"][s]) for s in ("mean", "std")}
    adim = meta.features["action"]["shape"][0]

    # action stats: for arm-delta training compute them on the delta targets (sampled frames)
    if B["arm_delta"]:
        rng = np.random.default_rng(0)
        samp = [ds[int(i)] for i in rng.choice(len(ds), size=min(4000, len(ds)), replace=False)]
        acts = torch.stack([s["action"] for s in samp])  # (N, chunk, D)
        stt = torch.stack([s["observation.state"] for s in samp])
        acts[..., :6] -= stt[:, None, :6]
        flat = acts.reshape(-1, adim)
        stats["action"] = {"mean": flat.mean(0), "std": flat.std(0).clamp_min(1e-4)}
    else:
        stats["action"] = {s: tens(st["action"][s]) for s in ("mean", "std")}

    task_list = sorted(meta.tasks.values()) if B["one_hot"] else []
    if task_list:
        in_feats["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (len(task_list),))
    task_by_index = {int(k): v for k, v in meta.tasks.items()}
    task_idx = {t: i for i, t in enumerate(task_list)}

    cfg = ACTConfig(chunk_size=a.chunk, n_action_steps=a.n_action_steps, input_features=in_feats,
                    output_features={"action": PolicyFeature(FeatureType.ACTION, (adim,))},
                    optimizer_lr=a.lr, optimizer_lr_backbone=a.lr / 5, device="cuda")
    policy = ACTPolicy(cfg, dataset_stats=stats).cuda().train()
    groups = [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": a.lr},
              {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": a.lr / 5}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total=a.steps, peak=a.lr, end=a.lr / 10))

    json.dump({"bench": a.bench, "repo": B["repo"], "images": B["images"], "tasks": task_list, "fps": fps,
               "chunk": a.chunk, "n_action_steps": a.n_action_steps, "arm_delta": B["arm_delta"], "batch": a.batch,
               "steps": a.steps, "lr": a.lr}, open(out / "meta.json", "w"), indent=1)

    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                         pin_memory=True, drop_last=True, persistent_workers=True)

    def save(step):
        policy.save_pretrained(out / "policy")
        json.dump({"step": step, "hours": (time.time() - t_start) / 3600}, open(out / "progress.json", "w"))

    step, log, done = 0, open(out / "train_log.jsonl", "a"), False
    while not done:
        for batch in loader:
            b = {k: batch[k].cuda(non_blocking=True) for k in B["images"]}
            b["observation.state"] = batch["observation.state"].cuda(non_blocking=True)
            act = batch["action"].clone()
            if B["arm_delta"]:
                act[..., :6] -= batch["observation.state"][:, None, :6]
            b["action"] = act.cuda(non_blocking=True)
            b["action_is_pad"] = batch["action_is_pad"].cuda(non_blocking=True)
            if task_list:
                ti = torch.tensor([task_idx[task_by_index[int(i)]] for i in batch["task_index"]])
                b["observation.environment_state"] = torch.nn.functional.one_hot(ti, len(task_list)).float().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, info = policy.forward(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                rec = {"step": step, "loss": float(loss), **info, "lr": sched.get_last_lr()[0],
                       "hours": round((time.time() - t_start) / 3600, 3)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 1000 == 0:
                    print(rec, flush=True)
            if step % a.save_every == 0:
                save(step)
            if step >= a.steps or (time.time() - t_start) / 3600 > a.max_hours:
                done = True
                break
    save(step)
    print("finished at step", step, flush=True)


if __name__ == "__main__":
    main()
