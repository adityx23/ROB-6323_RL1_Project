"""Check one generated part: color labels match every episode, the colored top cube ends on top, yield per color
pair, plus a contact sheet (first / last front frame of a few episodes with their prompts).

    python inspect_part.py <part.npz> [--sheet out.png] [--n 12]
"""

import argparse
import collections
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_colored import cube_key, cube_xyz_from_qpos  # noqa: E402

TASK = "block-double-task2"


def episode_labels(path):
    """Per-episode dicts (metadata + color label + order check), in file order."""
    lab = json.load(open(path.replace(".npz", "-colors.json")))
    addrs, labels = lab["cube_qpos_addrs"], lab["episodes"]
    meta = pickle.load(open(path.replace(".npz", "-metadata.pkl"), "rb"))["episodes"]
    qpos = np.load(path)["qpos"]
    out = []
    for e in meta:
        lb = labels.get(cube_key(cube_xyz_from_qpos(qpos[e["start"]], addrs)))
        z_end = cube_xyz_from_qpos(qpos[e["end"]], addrs)[:, 2]
        actual_top = int(np.argmax(z_end))
        order_ok = lb is not None and actual_top == lb["top"]
        out.append(dict(e, label=lb, actual_top=actual_top, order_ok=bool(order_ok),
                        usable=bool(e["success"] and order_ok)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    eps = episode_labels(a.part)
    n = len(eps)
    succ = [e for e in eps if e["success"]]
    print(f"episodes {n}  labeled {sum(e['label'] is not None for e in eps)}/{n}  "
          f"oracle success {len(succ)}/{n} = {len(succ) / n:.1%}")
    print(f"right cube on top, among successes: {sum(e['order_ok'] for e in succ)}/{len(succ)}  "
          f"-> usable {sum(e['usable'] for e in eps)}/{n} = {sum(e['usable'] for e in eps) / n:.1%}")
    pairs = collections.Counter((e["label"]["top_color"], e["label"]["bottom_color"]) for e in eps if e["usable"])
    print("usable per (top, bottom):", dict(sorted(pairs.items())))
    lens = [e["length"] for e in eps if e["usable"]]
    print(f"usable length min/mean/max {min(lens)}/{np.mean(lens):.0f}/{max(lens)}; "
          f"speed {np.min([e['speed'] for e in eps]):.2f}-{np.max([e['speed'] for e in eps]):.2f}")
    if a.sheet:
        import cv2

        from rlft_bench.data.ocb_import import load_ocb

        ep = load_ocb([a.part], TASK, successful_only=False, max_episodes=a.n)
        tiles = []
        for i in range(min(a.n, ep.num_episodes)):
            d, lb = ep.episode(i), eps[i]["label"]
            row = [cv2.resize(d["images"][j, c], (192, 192), interpolation=cv2.INTER_AREA)
                   for j in (0, -1) for c in (0, 2)]
            row = np.concatenate(row, 1).copy()
            txt = f"{i}: {lb['prompt'] if lb else 'NO LABEL'} | succ={eps[i]['success']} top_ok={eps[i]['order_ok']}"
            cv2.putText(row, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(row)
        sheet = np.concatenate(tiles, 0)
        cv2.imwrite(a.sheet, sheet[..., ::-1])
        print("sheet:", a.sheet, sheet.shape)


if __name__ == "__main__":
    main()
