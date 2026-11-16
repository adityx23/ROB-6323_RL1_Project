"""Export regenerated OCBench block-double-task2 (cube stacking) visual demos to LeRobot v2.1, streaming part by part.

Same frame format as overnight/export_ocb_lerobot.py (local/ocb_stack): front + wrist resized to 128x128 (INTER_AREA),
observation.state = 19-d proprio, action = absolute joint targets (6 arm + gripper) at every image step (10 fps).
Differences: one part file in memory at a time (the old exporter held every part, ~460 GB for 3k demos), and no
GPU sim: the JointSpace comes from OCB_JOINT_SCALE + the arm ctrlrange saved in ocb_stack/ocb_meta.json.

    python export_3k.py export [--num 3000] [--repo local/ocb_stack_3k]
    python export_3k.py check  [--episodes 3]   # re-derive ocb_stack's first episodes, compare to the saved ones
"""

import os

# Nothing here renders (images come from the part files); the OCBench CPU env that load_ocb builds only needs a
# GL backend that loads lazily. run.sh forces EGL, which CPU nodes lack, so switch to glfw before mujoco loads.
os.environ["MUJOCO_GL"] = "glfw"
os.environ.pop("PYOPENGL_PLATFORM", None)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rlft_bench.actions.spaces import OCB_JOINT_SCALE, JointSpace, relabel_episode_abs  # noqa: E402
from rlft_bench.data.episodes import PROPRIO_DIM  # noqa: E402
from rlft_bench.data.ocb_import import load_ocb  # noqa: E402

TASK = "block-double-task2"
PROMPT = "stack the two cubes"
CAMS = {"front": 0, "wrist": 2}
LEROBOT = pathlib.Path("/scratch/aap9348/rlft-bench/cache/huggingface/lerobot")
PARTS_GLOB = f"/scratch/aap9348/rlft-bench/data/ocbench/regen/visual-{TASK}-v0/i5/train-p*.npz"
OLD_META = json.load(open(LEROBOT / "local/ocb_stack/ocb_meta.json"))
JS = JointSpace(scale=OCB_JOINT_SCALE.copy(), arm_low=np.array(OLD_META["arm_low"]), arm_high=np.array(OLD_META["arm_high"]))


def parts():
    ps = sorted(p for p in glob.glob(PARTS_GLOB) if not p.endswith("-val.npz"))
    assert ps, "no regenerated parts found"
    return ps


def frames(d):
    """Yield the LeRobot frames of one episode dict from Episodes.episode()."""
    T = int(d["length"])
    acts, _ = relabel_episode_abs(JS, d["actions_rel"], d["q"][:-1])
    for j, t in enumerate(d["obs_t"]):
        if t >= T:
            continue
        fr = {f"observation.images.{c}": cv2.resize(d["images"][j, ci], (128, 128), interpolation=cv2.INTER_AREA)
              for c, ci in CAMS.items()}
        fr["observation.state"] = d["state"][j, :PROPRIO_DIM].astype(np.float32)
        fr["action"] = acts[t].astype(np.float32)
        yield fr


def export(num: int, repo: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = LEROBOT / repo
    if root.exists():
        shutil.rmtree(root)
    feats = {f"observation.images.{c}": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]}
             for c in CAMS}
    feats["observation.state"] = {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None}
    feats["action"] = {"dtype": "float32", "shape": (7,), "names": None}
    ds = None
    done, n_fr, lens, used, t0 = 0, 0, [], [], time.time()
    for p in parts():
        if done >= num:
            break
        ep = load_ocb([p], TASK, successful_only=True, max_episodes=num - done)
        if ds is None:
            fps = int(round(50 / ep.obs_interval))
            ds = LeRobotDataset.create(repo_id=repo, fps=fps, root=root, features=feats, use_videos=False,
                                       image_writer_threads=16)
        for e in range(ep.num_episodes):
            rows = 0
            for fr in frames(ep.episode(e)):
                fr["task"] = PROMPT
                ds.add_frame(fr)
                rows += 1
            ds.save_episode()
            n_fr += rows
            lens.append(rows)
        done += ep.num_episodes
        used.append([pathlib.Path(p).name, int(ep.num_episodes)])
        del ep
        print(f"{pathlib.Path(p).name}: total {done}/{num} episodes, {n_fr} frames, {time.time() - t0:.0f}s", flush=True)
    assert done >= num, f"only {done} successful demos (< {num})"
    json.dump({"prompt": PROMPT, "episodes": done, "frames": n_fr, "fps": fps,
               "rows_min_mean_max": [min(lens), float(np.mean(lens)), max(lens)],
               "arm_low": JS.arm_low.tolist(), "arm_high": JS.arm_high.tolist(), "parts": used},
              open(root / "ocb_meta.json", "w"), indent=1)
    print(f"exported {done} episodes, {n_fr} frames at {fps} fps -> {root} in {time.time() - t0:.0f}s", flush=True)


def check(n: int):
    """Regression check against local/ocb_stack (made by the GPU-sim exporter from the same first part)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    old = LeRobotDataset("local/ocb_stack", root=LEROBOT / "local/ocb_stack")
    ep = load_ocb([parts()[0]], TASK, successful_only=True, max_episodes=n)
    for e in range(n):
        new = list(frames(ep.episode(e)))
        lo, hi = int(old.episode_data_index["from"][e]), int(old.episode_data_index["to"][e])
        assert hi - lo == len(new), (e, hi - lo, len(new))
        a_old = np.stack([old.hf_dataset[i]["action"].numpy() for i in range(lo, hi)])
        s_old = np.stack([old.hf_dataset[i]["observation.state"].numpy() for i in range(lo, hi)])
        im_old = np.asarray(old[lo]["observation.images.front"].numpy() * 255).round().astype(np.uint8)
        im_old = np.moveaxis(im_old, 0, -1) if im_old.shape[0] == 3 else im_old
        da = np.abs(a_old - np.stack([f["action"] for f in new])).max()
        dst = np.abs(s_old - np.stack([f["observation.state"] for f in new])).max()
        dim = np.abs(im_old.astype(int) - new[0]["observation.images.front"].astype(int)).max()
        print(f"episode {e}: rows {len(new)}  max|d action| {da:.2e}  max|d state| {dst:.2e}  max|d front img0| {dim}")
        assert da < 1e-6 and dst < 1e-6 and dim <= 1, "mismatch with the overnight export"
    print("check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "check"])
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--repo", default="local/ocb_stack_3k")
    ap.add_argument("--episodes", type=int, default=3)
    a = ap.parse_args()
    export(a.num, a.repo) if a.mode == "export" else check(a.episodes)
