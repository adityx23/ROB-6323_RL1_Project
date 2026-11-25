"""Regenerate OCBench block-double-task2 (stack two cubes) visual demos with per-episode cube colors and wider
start-state / trajectory diversity. Leaves rlft-bench and OCBench untouched: it reuses rlft-bench's generator hooks
(rlft_bench/third_party/ocbench_streaming_data.HOOKS) and swaps in a per-episode-color renderer.

Colors and rules come from color_rules.json: every episode samples one allowed (top, bottom) color pair. Right after
each scripted-policy reset, the planned stacking order is read from the controller (goal_order[w, 0] = bottom cube,
goal_order[w, 1] = top cube) and the cubes are colored so the top cube gets the top color. The images are rendered
one world at a time on a CPU MuJoCo copy (OCBench's own path); this script repaints the cubes right before each
render. Success recoloring (OCBench turns a placed cube pale) is disabled, so color never leaks success.

Per-episode labels go to <name>-colors.json, keyed by the episode's initial cube positions (exact float32 bytes),
which export_diverse.py matches against each saved episode's first qpos row. Episodes whose final state has the wrong
cube on top (the scripted policy's built-in mistakes) are dropped at export.

    python gen_colored.py --episodes 110 --seed 101 --name train-p000 [--coverage 1.3 --speed 0.5 2.0 --dart 0.05]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.environ.get("RLFT_ROOT", "."))  # rlft_bench

import numpy as np  # noqa: E402

TASK = "block-double-task2"
ENV = f"visual-{TASK}-v0"
RULES_PATH = pathlib.Path(__file__).with_name("color_rules.json")


def load_rules(path=RULES_PATH):
    r = json.load(open(path))
    pairs = [tuple(p) for p in r["fixed"]] + [tuple(p) for p in r["free"]] + [tuple(p[::-1]) for p in r["free"]]
    never = {frozenset(p) for p in r["never"]}
    assert not any(frozenset(p) in never for p in pairs), "a pair is both allowed and never-together"
    assert len(set(pairs)) == len(pairs), "duplicate pairs"
    return r, pairs


def cube_key(cube_xyz: np.ndarray) -> str:
    return np.ascontiguousarray(cube_xyz, dtype=np.float32).tobytes().hex()


def cube_xyz_from_qpos(qpos_rows: np.ndarray, addrs) -> np.ndarray:
    return np.stack([qpos_rows[..., a:a + 3] for a in addrs], axis=-2)  # (..., n_cubes, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=110)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=os.environ.get("RLFT_DATA", "data/ocbench"))
    ap.add_argument("--out-dir", default="regen-diverse")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--coverage", type=float, default=1.3)
    ap.add_argument("--speed", type=float, nargs=2, default=(0.5, 2.0))
    ap.add_argument("--dart", type=float, default=0.05)
    ap.add_argument("--max-parallel", type=int, default=55)
    a = ap.parse_args()

    from rlft_bench.data.regenerate import _scale_box
    from rlft_bench.third_party import ocbench_streaming_data as sd

    rules, pairs = load_rules()
    rgb = {c: np.array(v, dtype=np.float64) for c, v in rules["colors"].items()}
    rng = np.random.default_rng(a.seed + 777)
    state = {"world_rgb": None, "addrs": None}  # world_rgb: (nworld, n_cubes, 3), refreshed after every reset
    labels = {}

    def paint(cpu, colors):
        for i, c in enumerate(colors):
            cpu._cube_colors[i, :3] = c
            cpu._cube_success_colors[i, :3] = c  # no pale "success" color
            for gid in cpu._cube_geom_ids_list[i]:
                cpu._model.geom(gid).rgba[:3] = c
            for gid in cpu._cube_target_geom_ids_list[i]:
                cpu._model.geom(gid).rgba[:3] = c

    def post_make_env(env):
        cpu = env.cpu_env
        if a.coverage != 1.0:
            for name in ("_arm_sampling_bounds", "_object_sampling_bounds", "_target_sampling_bounds"):
                if hasattr(cpu, name):
                    box = getattr(cpu, name)
                    ws = cpu._workspace_bounds[:, : box.shape[1]]
                    setattr(cpu, name, np.clip(_scale_box(box, a.coverage), ws[0], ws[1]))
        cpu._cube_colors = cpu._cube_colors.copy()
        cpu._cube_success_colors = cpu._cube_success_colors.copy()

        def get_pixel_observation(world_ids=None, *, arrays=None):
            world_ids = range(env._nworld) if world_ids is None else world_ids
            arrays = env._get_arrays() if arrays is None else arrays
            images = []
            for w in world_ids:
                env.copy_world_to_cpu_env(w, arrays=arrays)
                paint(cpu, state["world_rgb"][w])
                images.append(cpu.get_pixel_observation())
            return np.stack(images)

        env.get_pixel_observation = get_pixel_observation

    def post_controller_reset(env, ctl):
        lo, hi = a.speed
        scale = getattr(ctl, "segment_dt_scale", 1.0)
        vals = rng.uniform(lo, hi, ctl.nworld).astype(np.float32) * scale
        ctl.wp.copy(ctl.speed_dt, ctl.wp.array(vals, dtype=ctl.wp.float32, device=ctl.device))
        ctl.wp.synchronize()
        order = ctl.goal_order.numpy()
        qpos = env.data.qpos.numpy()
        addrs = [int(x) for x in env._object_qpos_addrs]
        n_cubes = len(addrs)
        state["addrs"] = addrs
        assert n_cubes == 2, n_cubes
        world_rgb = np.zeros((ctl.nworld, n_cubes, 3))
        for w in range(ctl.nworld):
            bottom, top = int(order[w, 0]), int(order[w, 1])
            top_c, bot_c = pairs[rng.integers(len(pairs))]
            names = [None] * n_cubes
            names[top], names[bottom] = top_c, bot_c
            world_rgb[w] = [rgb[n] for n in names]
            key = cube_key(cube_xyz_from_qpos(qpos[w], addrs))
            labels[key] = dict(colors=names, top=top, bottom=bottom, top_color=top_c, bottom_color=bot_c,
                               prompt=rules["prompt"].format(top=top_c, bottom=bot_c))
        state["world_rgb"] = world_rgb

    def action_noise(action, step):
        return np.clip(action + rng.normal(0, a.dart, action.shape), -1, 1)

    sd.HOOKS.update(post_make_env=post_make_env, post_controller_reset=post_controller_reset,
                    action_noise=action_noise if a.dart > 0 else None)
    t0 = time.time()
    try:
        train, val, metrics, train_meta, val_meta = sd.generate_mjwarp_datasets(
            ENV, collection_episodes=a.episodes, collection_val_episodes=0, root_seed=a.seed, collection_idx=0,
            max_parallel_episodes=a.max_parallel, return_metadata=True, observation_interval=a.interval)
    finally:
        sd.HOOKS.update(post_make_env=None, post_controller_reset=None, action_noise=None)

    out = pathlib.Path(a.root) / a.out_dir / ENV / f"i{a.interval}" / f"{a.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **train)
    knobs = dict(coverage=a.coverage, speed=list(a.speed), dart=a.dart, interval=a.interval, seed=a.seed,
                 rules=str(RULES_PATH))
    meta = dict(train_meta, rlft_knobs=knobs)
    with open(str(out).replace(".npz", "-metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)
    json.dump({"cube_qpos_addrs": state["addrs"], "episodes": labels}, open(str(out).replace(".npz", "-colors.json"), "w"))
    succ = [bool(e["success"]) for e in train_meta["episodes"]]
    print(json.dumps({"path": str(out), "episodes": len(succ), "success_rate": float(np.mean(succ)),
                      "labels": len(labels), "seconds": round(time.time() - t0), "knobs": knobs,
                      "MB_per_episode": out.stat().st_size / len(succ) / 1e6}, indent=1), flush=True)


if __name__ == "__main__":
    main()
