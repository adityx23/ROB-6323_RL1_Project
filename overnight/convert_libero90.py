"""LIBERO-90 bowl-stacking demos (4 tasks x 50, HDF5) -> LeRobot v2.1 dataset local/libero90_stack.

Features: observation.images.image / observation.images.wrist (128x128x3, as rendered: no flip, so the eval
client must not flip either), observation.state (8 = ee_pos 3 + ee_ori axis-angle 3 + gripper qpos 2),
action (7, LIBERO delta EEF + gripper). fps 20 (LIBERO control rate). No-op steps (zero motion, unchanged
gripper) are dropped, as in the OpenVLA/openpi LIBERO data.
"""

import os

import glob
import json
import pathlib
import shutil

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

RAW = os.environ["RLFT_ROOT"] + "/data/libero90_raw/libero_90"
ROOT = pathlib.Path(os.environ["RLFT_ROOT"] + "/cache/huggingface/lerobot/local/libero90_stack")
TASK_IDS = {"KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle": 16,
            "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl": 17,
            "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray": 63,
            "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray": 64}


def is_noop(a, prev):
    return np.linalg.norm(a[:-1]) < 1e-4 and prev is not None and a[-1] == prev[-1]


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    feats = {"observation.images.image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
             "observation.images.wrist": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
             "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
             "action": {"dtype": "float32", "shape": (7,), "names": None}}
    ds = LeRobotDataset.create(repo_id="local/libero90_stack", fps=20, root=ROOT, features=feats, use_videos=False,
                               image_writer_threads=8)
    info, n_ep, n_fr, dropped = {}, 0, 0, 0
    for f in sorted(glob.glob(f"{RAW}/*stack*_demo.hdf5")):
        name = pathlib.Path(f).name.replace("_demo.hdf5", "")
        prompt = name.split("SCENE", 1)[1].split("_", 1)[1].replace("_", " ")
        info[prompt] = TASK_IDS[name]
        with h5py.File(f, "r") as h:
            for k in sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1])):
                d = h["data"][k]
                o = d["obs"]
                acts = d["actions"][()]
                state = np.concatenate([o["ee_pos"][()], o["ee_ori"][()], o["gripper_states"][()]], 1).astype(np.float32)
                img, wr = o["agentview_rgb"][()], o["eye_in_hand_rgb"][()]
                prev = None
                for t in range(len(acts)):
                    if is_noop(acts[t], prev):
                        dropped += 1
                        continue
                    prev = acts[t]
                    ds.add_frame({"observation.images.image": img[t], "observation.images.wrist": wr[t],
                                  "observation.state": state[t], "action": acts[t].astype(np.float32), "task": prompt})
                    n_fr += 1
                ds.save_episode()
                n_ep += 1
        print(f"{prompt}: done ({n_ep} episodes so far)", flush=True)
    json.dump({"prompt_to_libero90_task_id": info, "episodes": n_ep, "frames": n_fr, "dropped_noops": dropped},
              open(ROOT / "l90_meta.json", "w"), indent=1)
    print(f"episodes {n_ep}, frames {n_fr}, dropped no-ops {dropped} -> {ROOT}")


if __name__ == "__main__":
    main()
