"""Replace the 3k cube-stacking dataset on Hugging Face with the diverse colored set (local/ocb_diverse_1k).

Steps: rename the dataset repo FROM -> TO (HF redirects the old name; skipped if TO exists), upload the diverse set
(hard-link staging + upload_large_folder, resumable), then delete every remote file that is not part of it (the
old 3k episodes and metadata) and check the remote file list equals the staged one. The repo keeps its visibility.
The 3k set stays on Torch at $RLFT_ROOT/cache/huggingface/lerobot/local/ocb_stack_3k.

    python upload_diverse_hf.py [--from-repo adityx23/ocb-stack-3k] [--repo adityx23/ocb-stack-diverse-1k] [--dry-run]
"""

import argparse
import collections
import json
import os
import pathlib
import shutil

R = pathlib.Path(os.environ["RLFT_ROOT"])
SRC = R / "cache/huggingface/lerobot/local/ocb_diverse_1k"
NORM = R / "runs/diverse/assets/pi05_ocbdiv1k/local/ocb_diverse_1k/norm_stats.json"
NORM_DST = "norm_stats/pi05_ocbdiv1k/local/ocb_diverse_1k/norm_stats.json"
RULES = pathlib.Path(__file__).with_name("color_rules.json")
STAGE = R / "runs/diverse/hf_data_stage"

README = """---
tags: [robotics, lerobot, ocbench, manipulation, language-conditioned]
---
# OCBench cube stacking, diverse colored set: {episodes} demos (LeRobot v2.1)

OCBench `block-double-task2` (stack two cubes, UR5e, MuJoCo Warp), regenerated with visual observations,
per-episode cube colors with stacking rules, and wider start-state / trajectory diversity. Each episode's task string
is its language prompt, which names the colors: "stack the red cube on top of the blue cube".

| | |
|---|---|
| Episodes | {episodes} demos, {frames} frames, {fps} fps; kept only if the scripted policy succeeded AND the prompted cube ended on top |
| Images | `observation.images.front`, `observation.images.wrist`: 128x128 RGB |
| State / action | `observation.state` 19-d proprio; `action` 7-d absolute joint targets (6 arm + gripper) |
| Diversity | start-area coverage 1.3x OCBench default, scripted speed U[0.5, 2.0], DART action noise 0.05 |

## Colors and rules (`color_rules.json`)

Colors: {colors}.

| Rule | Pairs (top on bottom) |
|---|---|
| Fixed order (only this order appears) | {fixed} |
| Never together (never in the same scene) | {never} |
| Either order | {free} |

Demos per (top, bottom) pair: {pairs}

`ocb_meta.json` holds per-episode labels (colors, top/bottom cube, prompt, source part). The quantile norm stats
openpi computed for the training config are in `norm_stats/pi05_ocbdiv1k/local/ocb_diverse_1k/norm_stats.json`. Use
the dataset as `local/ocb_diverse_1k` under `$HF_LEROBOT_HOME`.

This repo previously held the 3000-demo single-prompt set ("stack the two cubes"); see the commit history.
"""


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(SRC, STAGE, copy_function=os.link, ignore=shutil.ignore_patterns(".cache"))
    (STAGE / NORM_DST).parent.mkdir(parents=True)
    shutil.copy(NORM, STAGE / NORM_DST)
    shutil.copy(RULES, STAGE / "color_rules.json")
    meta = json.load(open(SRC / "ocb_meta.json"))
    rules = json.load(open(RULES))
    on = lambda ps: ", ".join(f"{a} on {b}" for a, b in ps)  # noqa: E731
    pairs = collections.OrderedDict(sorted(meta["pair_counts"].items()))
    (STAGE / "README.md").write_text(README.format(
        episodes=meta["episodes"], frames=f"{meta['frames']:,}", fps=meta["fps"], colors=", ".join(rules["colors"]),
        fixed=on(rules["fixed"]), never=", ".join(f"{a} + {b}" for a, b in rules["never"]),
        free=", ".join(f"{a} / {b}" for a, b in rules["free"]),
        pairs=", ".join(f"{k}: {v}" for k, v in pairs.items())))
    files = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {len(files)} files, {sum((STAGE / f).stat().st_size for f in files) / 1e9:.1f} GB", flush=True)
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-repo", default="adityx23/ocb-stack-3k")
    ap.add_argument("--repo", default="adityx23/ocb-stack-diverse-1k")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = stage()
    if a.dry_run:
        raise SystemExit(0)
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi()
    if not api.repo_exists(a.repo, repo_type="dataset"):
        if api.repo_exists(a.from_repo, repo_type="dataset"):
            api.move_repo(a.from_repo, a.repo, repo_type="dataset")
            print(f"renamed {a.from_repo} -> {a.repo}", flush=True)
        else:
            api.create_repo(a.repo, repo_type="dataset", private=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(STAGE), num_workers=8)
    keep = set(files) | {".gitattributes"}
    stale = [f for f in api.list_repo_files(a.repo, repo_type="dataset") if f not in keep]
    for i in range(0, len(stale), 500):
        api.create_commit(a.repo, repo_type="dataset",
                          operations=[CommitOperationDelete(f) for f in stale[i:i + 500]],
                          commit_message=f"Remove the previous 3k-demo files ({i + 1}-{min(i + 500, len(stale))} of {len(stale)})")
    remote = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset") if f != ".gitattributes")
    print(f"removed {len(stale)} old files; remote now {len(remote)} files (staged {len(files)}) in "
          f"https://huggingface.co/datasets/{a.repo}", flush=True)
    assert remote == files, "remote file list differs from the staged set: rerun to resume"
    shutil.rmtree(STAGE)
