"""Export the diverse colored cube-stacking demos to LeRobot v2.1 (local/ocb_diverse_1k), one part at a time.

Same frame format as ladder/export_3k.py (front + wrist 128x128, 19-d proprio state, 7-d absolute joint targets at
10 fps). Differences: each episode's task string is its own colored prompt ("stack the red cube on top of the blue
cube"), and only usable episodes are kept: oracle success AND the prompted top cube actually on top at the end.
Per-episode labels (colors, top/bottom, prompt, source part/index) go to ocb_meta.json.

    python export_diverse.py [--num 1000] [--repo local/ocb_diverse_1k]
"""

import os

os.environ["MUJOCO_GL"] = "glfw"  # CPU node: nothing renders here (see ladder/export_3k.py)
os.environ.pop("PYOPENGL_PLATFORM", None)

import argparse  # noqa: E402
import collections  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.environ["RLFT_ROOT"])
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "ladder"))

from export_3k import CAMS, JS, LEROBOT, TASK, frames  # noqa: E402
from inspect_part import episode_labels  # noqa: E402
from rlft_bench.data.episodes import PROPRIO_DIM  # noqa: E402
from rlft_bench.data.ocb_import import load_ocb  # noqa: E402

PARTS_GLOB = f"{os.environ['RLFT_ROOT']}/data/ocbench/regen-diverse/visual-{TASK}-v0/i5/train-p*.npz"


def export(num: int, repo: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = LEROBOT / repo
    if root.exists():
        shutil.rmtree(root)
    feats = {f"observation.images.{c}": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]}
             for c in CAMS}
    feats["observation.state"] = {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None}
    feats["action"] = {"dtype": "float32", "shape": (7,), "names": None}
    ds, fps, rows_per_ep, kept, t0 = None, None, [], [], time.time()
    n_seen = n_success = 0
    for p in sorted(glob.glob(PARTS_GLOB)):
        if len(kept) >= num:
            break
        labels = episode_labels(p)
        n_seen += len(labels)
        n_success += sum(e["success"] for e in labels)
        ep = load_ocb([p], TASK, successful_only=False)
        assert ep.num_episodes == len(labels), (p, ep.num_episodes, len(labels))
        if ds is None:
            fps = int(round(50 / ep.obs_interval))
            ds = LeRobotDataset.create(repo_id=repo, fps=fps, root=root, features=feats, use_videos=False,
                                       image_writer_threads=16)
        for i, lab in enumerate(labels):
            if not lab["usable"] or len(kept) >= num:
                continue
            d = ep.episode(i)
            assert int(d["length"]) == lab["length"], (p, i)
            rows = 0
            for fr in frames(d):
                fr["task"] = lab["label"]["prompt"]
                ds.add_frame(fr)
                rows += 1
            ds.save_episode()
            rows_per_ep.append(rows)
            kept.append(dict(part=pathlib.Path(p).name, index=i, **{k: lab["label"][k] for k in
                             ("prompt", "top_color", "bottom_color", "colors", "top", "bottom")}))
        del ep
        print(f"{pathlib.Path(p).name}: kept {len(kept)}/{num}, {sum(rows_per_ep)} frames, {time.time() - t0:.0f}s",
              flush=True)
    assert len(kept) >= num, f"only {len(kept)} usable demos (< {num}): generate more parts"
    pairs = collections.Counter(f"{k['top_color']} on {k['bottom_color']}" for k in kept)
    json.dump({"episodes": len(kept), "frames": sum(rows_per_ep), "fps": fps,
               "rows_min_mean_max": [min(rows_per_ep), float(np.mean(rows_per_ep)), max(rows_per_ep)],
               "generated_seen": n_seen, "oracle_success": n_success, "pair_counts": dict(sorted(pairs.items())),
               "rules": json.load(open(pathlib.Path(__file__).with_name("color_rules.json"))),
               "arm_low": JS.arm_low.tolist(), "arm_high": JS.arm_high.tolist(), "per_episode": kept},
              open(root / "ocb_meta.json", "w"), indent=1)
    print(f"exported {len(kept)} episodes, {sum(rows_per_ep)} frames -> {root} in {time.time() - t0:.0f}s", flush=True)
    print("pairs:", dict(sorted(pairs.items())), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--repo", default="local/ocb_diverse_1k")
    a = ap.parse_args()
    export(a.num, a.repo)
