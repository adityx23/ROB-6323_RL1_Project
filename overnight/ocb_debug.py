"""Debug ACT's 0% on OCBench stacking. Writes OUT/report.txt and OUT/*.png.

A. Offline: ACT (through act_serve.ActServer, the eval code path) on training frames vs dataset actions.
B. Inputs: eval-rendered images/proprio at reset vs dataset episode starts (side-by-side png + stats).
C. Closed loop: 4 worlds, 600 control steps with ACT; log commanded target - q and achieved motion.
"""

import os

import json
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from act_serve import ActServer  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from rlft_bench.envs.ocbench_adapter import OCBenchAdapter  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
rep = open(OUT / "report.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    rep.write(s + "\n")
    rep.flush()


def u8(img_chw):
    return (img_chw.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


srv = ActServer(os.environ["RLFT_ROOT"] + "/runs/overnight/act_ocbs")
ds = LeRobotDataset("local/ocb_stack", delta_timestamps={"action": [i / 10 for i in range(10)]})
log("dataset frames", len(ds), "episodes", ds.num_episodes)

# ---- A. offline prediction error
rng = np.random.default_rng(0)
errs, hold = [], []
for i in rng.choice(len(ds), 300, replace=False):
    x = ds[int(i)]
    obs = {"observation/front": u8(x["observation.images.front"]), "observation/wrist": u8(x["observation.images.wrist"]),
           "observation/state": x["observation.state"].numpy(), "prompt": "stack the two cubes"}
    pred = srv.infer(obs)["actions"]
    gt = x["action"].numpy()
    valid = ~x["action_is_pad"].numpy()
    errs.append(np.abs(pred - gt)[valid].mean(0))
    hold.append(np.abs(np.concatenate([np.repeat(x["observation.state"].numpy()[None, :6], 10, 0), gt[:, 6:]], 1) - gt)[valid].mean(0))
log("A. ACT |pred - gt| per dim (rad / gripper frac):", np.mean(errs, 0).round(4))
log("   'hold current joints' baseline |q - gt|     :", np.mean(hold, 0).round(4))

# ---- B. inputs at reset vs dataset starts
env = OCBenchAdapter("block-double-task2", num_envs=4, visual=True, seed=777)
env.reset(seeds=np.arange(777, 781, dtype=np.uint32))
obs = env.observe(render=True)
small = lambda im: cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)  # noqa: E731
starts = ds.episode_data_index["from"][:20].tolist()
ds_states = np.stack([ds[int(s)]["observation.state"].numpy() for s in starts])
log("B. proprio at reset (env)    mean", obs["proprio"].mean(0).round(3))
log("   proprio at ep start (data) mean", ds_states.mean(0).round(3))
log("   images env dtype/shape", obs["images"].dtype, obs["images"].shape, "range", obs["images"].min(), obs["images"].max())
x0 = ds[int(starts[0])]
row_env = np.concatenate([small(obs["images"][0, 0]), small(obs["images"][0, 2])], 1)
row_ds = np.concatenate([u8(x0["observation.images.front"]), u8(x0["observation.images.wrist"])], 1)
cv2.imwrite(str(OUT / "B_env_top_vs_data_bottom.png"), cv2.cvtColor(np.concatenate([row_env, row_ds], 0), cv2.COLOR_RGB2BGR))

# ---- C. closed loop
plans = [[] for _ in range(4)]
target = np.zeros((4, 7), np.float32)
for row in range(120):  # 120 rows x 5 steps = 600 control steps
    obs = env.observe(render=True)
    q0 = env.measured_q()
    for i in range(4):
        if not plans[i]:
            el = {"observation/front": small(obs["images"][i, 0]), "observation/wrist": small(obs["images"][i, 2]),
                  "observation/state": obs["proprio"][i].astype(np.float32), "prompt": "stack the two cubes"}
            plans[i] = list(srv.infer(el)["actions"][:5])
        target[i] = plans[i].pop(0)
    for _ in range(5):
        _, _, term, trunc, info = env.step(target, absolute=True)
    q1 = env.measured_q()
    if row % 10 == 0:
        log(f"C. row {row:3d}: |target-q0| arm {np.abs(target[:, :6] - q0[:, :6]).mean():.4f}  moved {np.abs(q1[:, :6] - q0[:, :6]).mean():.4f}"
            f"  grip tgt {target[:, 6].round(2)} q {q1[:, 6].round(2)}  proprio[:6]-q {np.abs(obs['proprio'][:, :6] - q0[:, :6]).max():.4f}")
    if row in (0, 40, 80, 119):
        cv2.imwrite(str(OUT / f"C_row{row}.png"), cv2.cvtColor(np.concatenate([small(obs["images"][0, 0]), small(obs["images"][0, 2])], 1), cv2.COLOR_RGB2BGR))
log("done")
