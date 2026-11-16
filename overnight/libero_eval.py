"""LIBERO eval client (runs in the py3.8 libero venv). Adapted from lair-openpi examples/libero/main.py.

    python libero_eval.py --policy NAME --suite libero_spatial --port 8000 --tasks 0,1,2 --trials 20 \
        --replan 5 --out RESULTS_DIR

Writes RESULTS_DIR/videos/<policy>/<suite>/t<task>_e<ep>_<success|failure>.mp4 (agentview | wrist, 10 fps)
and appends one JSON line per episode to RESULTS_DIR/results.jsonl.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # sim clients: single-threaded math libs (several clients share a node)

import argparse
import collections
import json
import math
import os
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as wcp

DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(q):
    q = q.copy()
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--suite-label", default=None)
    a = ap.parse_args()
    np.random.seed(a.seed)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task_ids = range(suite.n_tasks) if a.tasks == "all" else [int(t) for t in a.tasks.split(",")]
    out = pathlib.Path(a.out)
    label = a.suite_label or a.suite
    vdir = out / "videos" / a.policy / label
    vdir.mkdir(parents=True, exist_ok=True)
    client = [None]

    def infer(el):
        for attempt in range(3):
            try:
                if client[0] is None:
                    client[0] = wcp.WebsocketClientPolicy(a.host, a.port)
                return client[0].infer(el)
            except Exception as e:  # noqa: BLE001  (dropped connection -> reconnect and retry)
                print(f"infer failed ({e!r}); reconnecting", flush=True)
                client[0] = None
                time.sleep(2)
        raise RuntimeError("policy server unreachable")
    max_steps = MAX_STEPS[a.suite]

    for tid in task_ids:
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.res, camera_widths=a.res, hard_reset=False)
        env.seed(a.seed)
        prompt = task.language
        for ep in range(a.trials):
            t0 = time.time()
            env.reset()
            plan = collections.deque()
            obs = env.set_init_state(init_states[ep])
            frames, done, t, err = [], False, 0, None
            while t < max_steps + 10:
                try:
                    if t < 10:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    # no flip: the LIBERO-90 HDF5 demos are stored as rendered
                    img = np.ascontiguousarray(obs["agentview_image"])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                    frames.append(np.concatenate([img, wrist], axis=1))
                    if not plan:
                        el = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                            "prompt": str(prompt),
                        }
                        chunk = infer(el)["actions"]
                        plan.extend(chunk[: a.replan])
                    obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
                    t += 1
                    if done:
                        break
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
                    break
            tag = "success" if done else "failure"
            vid = vdir / f"t{tid}_e{ep}_{tag}.mp4"
            if frames:  # 20 Hz control -> keep every 2nd frame, 10 fps (real time)
                imageio.mimwrite(vid, frames[::2], fps=10, quality=7, macro_block_size=8)
            rec = {"policy": a.policy, "bench": "libero", "suite": label, "task_id": tid, "task": prompt, "episode": ep,
                   "success": bool(done), "steps": t, "secs": round(time.time() - t0, 1), "error": err,
                   "video": str(vid.relative_to(out))}
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        env.close()


if __name__ == "__main__":
    main()
