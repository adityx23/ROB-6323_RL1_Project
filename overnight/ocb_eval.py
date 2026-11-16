"""OCBench block-double-task2 (cube stacking) eval client (rlft-bench .venv), talking to a policy server.

    python ocb_eval.py --policy NAME --port 8000 --episodes 50 --replan-rows 5 --out RESULTS_DIR

Each policy row is an absolute joint target held for 5 control steps (the training image interval).
Worlds run batched in one OCBenchAdapter; the server is queried once per live world per decision.
Videos: front | wrist at every row (10 fps). One JSON line per episode in RESULTS_DIR/results.jsonl.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # sim clients: single-threaded math libs (several clients share a node)

import argparse
import json
import pathlib
import time

import cv2
import imageio
import numpy as np
from openpi_client import websocket_client_policy as wcp

from rlft_bench.envs.ocbench_adapter import OCBenchAdapter

TASK = "block-double-task2"
PROMPT = "stack the two cubes"
SUITE = "ocb_stack"


def small(x):
    return cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)
HOLD = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--replan-rows", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    vdir = out / "videos" / a.policy / SUITE
    vdir.mkdir(parents=True, exist_ok=True)
    client = [wcp.WebsocketClientPolicy("127.0.0.1", a.port)]

    def infer(el):
        for attempt in range(3):
            try:
                return client[0].infer(el)
            except Exception as e:  # noqa: BLE001
                print(f"infer failed ({e!r}); reconnecting", flush=True)
                time.sleep(2)
                client[0] = wcp.WebsocketClientPolicy("127.0.0.1", a.port)
        raise RuntimeError("policy server unreachable")

    n = a.episodes
    env = OCBenchAdapter(TASK, num_envs=n, visual=True, seed=a.seed)
    seeds = np.arange(a.seed, a.seed + n, dtype=np.uint32)
    env.reset(seeds=seeds)
    max_steps = env.max_episode_steps
    live = np.ones(n, bool)
    success = np.zeros(n, bool)
    length = np.zeros(n, np.int64)
    frames = [[] for _ in range(n)]
    plans = [[] for _ in range(n)]
    target = np.zeros((n, 7), np.float32)
    t0, t = time.time(), 0
    while live.any() and t < max_steps:
        obs = env.observe(render=True)
        imgs, prop = obs["images"], obs["proprio"]
        for i in np.where(live)[0]:
            front, wrist = small(imgs[i, 0]), small(imgs[i, 2])
            frames[i].append(np.concatenate([front, wrist], axis=1))
            if not plans[i]:
                el = {"observation/front": front, "observation/wrist": wrist,
                      "observation/state": prop[i].astype(np.float32), "prompt": PROMPT}
                chunk = np.asarray(infer(el)["actions"])
                plans[i] = list(chunk[: a.replan_rows])
            target[i] = plans[i].pop(0)
        # dead worlds hold their measured position
        q = env.measured_q()
        target[~live] = q[~live, :7]
        for _ in range(HOLD):
            _, _, term, trunc, info = env.step(target, absolute=True)
            succ = np.asarray(info.get("success", np.zeros(n, bool)), bool)
            success |= live & succ
            length += live
            live &= ~(np.asarray(term, bool) | np.asarray(trunc, bool) | succ)
            t += 1
            if not live.any():
                break
        if t % 250 < HOLD:
            print(f"t={t} live={live.sum()} success={success.sum()} {time.time() - t0:.0f}s", flush=True)

    for i in range(n):
        tag = "success" if success[i] else "failure"
        vid = vdir / f"t0_e{a.seed - 12345 + i}_{tag}.mp4"
        imageio.mimwrite(vid, frames[i], fps=10, quality=7, macro_block_size=8)
        rec = {"policy": a.policy, "bench": "ocb", "suite": SUITE, "task_id": 0, "task": PROMPT,
               "episode": int(a.seed - 12345 + i), "success": bool(success[i]), "steps": int(length[i]), "seed": int(seeds[i]),
               "video": str(vid.relative_to(out))}
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"{a.policy}: success {success.mean():.3f} ({success.sum()}/{n}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
