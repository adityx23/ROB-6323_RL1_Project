"""Export regenerated OCBench block-double-task2 (cube stacking) visual demos to LeRobot v2.1: local/ocb_stack.

Shared-recipe format (matches local/libero90_stack): 200 successful demos, two views (front, wrist) resized to
128x128, observation.state = 19-d proprio (joint pos 0:6 ...), action = absolute joint targets (6 arm + gripper
opening) at every image step (interval 5 -> 10 fps). Training turns arm targets into deltas from state[:6].
"""

import os

import glob
import json
import pathlib
import shutil

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rlft_bench.actions.spaces import JointSpace, relabel_episode_abs
from rlft_bench.data.episodes import PROPRIO_DIM
from rlft_bench.data.ocb_import import load_ocb
from rlft_bench.envs.ocbench_adapter import OCBenchAdapter

TASK = "block-double-task2"
PROMPT = "stack the two cubes"
NUM = 200
CAMS = {"front": 0, "wrist": 2}
ROOT = pathlib.Path(os.environ["RLFT_ROOT"] + "/cache/huggingface/lerobot/local/ocb_stack")

paths = sorted(glob.glob(os.environ["RLFT_ROOT"] + f"/data/ocbench/regen/visual-{TASK}-v0/i5/train-p*.npz"))
paths = [p for p in paths if not p.endswith("-val.npz")]
assert paths, "no regenerated parts found"
print("parts:", paths, flush=True)
ep = load_ocb(paths, TASK, successful_only=True)
print(f"successful episodes: {ep.num_episodes}, images {ep.images.shape}", flush=True)
assert ep.num_episodes >= NUM, f"only {ep.num_episodes} successful demos (< {NUM})"

env = OCBenchAdapter(TASK, num_envs=1, visual=True)
js = JointSpace.from_adapter(env)
if ROOT.exists():
    shutil.rmtree(ROOT)
feats = {f"observation.images.{c}": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]}
         for c in CAMS}
feats["observation.state"] = {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None}
feats["action"] = {"dtype": "float32", "shape": (7,), "names": None}
fps = int(round(50 / ep.obs_interval))
ds = LeRobotDataset.create(repo_id="local/ocb_stack", fps=fps, root=ROOT, features=feats, use_videos=False,
                           image_writer_threads=8)
n_fr, lens = 0, []
for e in range(NUM):
    d = ep.episode(e)
    T = int(d["length"])
    acts, _ = relabel_episode_abs(js, d["actions_rel"], d["q"][:-1])
    rows = 0
    for j, t in enumerate(d["obs_t"]):
        if t >= T:
            continue
        fr = {f"observation.images.{c}": cv2.resize(d["images"][j, ci], (128, 128), interpolation=cv2.INTER_AREA)
              for c, ci in CAMS.items()}
        fr["observation.state"] = d["state"][j, :PROPRIO_DIM].astype(np.float32)
        fr["action"] = acts[t].astype(np.float32)
        fr["task"] = PROMPT
        ds.add_frame(fr)
        rows += 1
    ds.save_episode()
    n_fr += rows
    lens.append(rows)
json.dump({"prompt": PROMPT, "episodes": NUM, "frames": n_fr, "fps": fps, "rows_min_mean_max": [min(lens), float(np.mean(lens)), max(lens)],
           "arm_low": js.arm_low.tolist(), "arm_high": js.arm_high.tolist()}, open(ROOT / "ocb_meta.json", "w"), indent=1)
print(f"exported {NUM} episodes, {n_fr} frames at {fps} fps -> {ROOT}", flush=True)
