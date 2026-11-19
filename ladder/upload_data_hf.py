"""Upload the OCBench cube-stacking training data to one Hugging Face dataset repo (private by default).

Repo layout: the LeRobot v2.1 dataset local/ocb_stack_3k at the root (data/, meta/, ocb_meta.json), the ladder's
norm stats under norm_stats/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json, plus README.md. Files are hard-linked
into a staging dir (same filesystem, no copy) and sent with upload_large_folder, which is resumable: rerunning
continues where it stopped. Token: HF_TOKEN (set by upload_data.sbatch). maxwell/setup.sh downloads this repo.

    python upload_data_hf.py [--repo adityx23/ocb-stack-3k] [--public] [--dry-run]
"""

import argparse
import os
import pathlib
import shutil

R = pathlib.Path(os.environ["RLFT_ROOT"])
SRC = R / "cache/huggingface/lerobot/local/ocb_stack_3k"
NORM = R / "runs/ladder/assets/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json"
NORM_DST = "norm_stats/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json"
STAGE = R / "runs/ladder/hf_data_stage"

README = """---
tags: [robotics, lerobot, ocbench, manipulation]
---
# OCBench cube stacking: 3000 scripted demos (LeRobot v2.1)

OCBench `block-double-task2` (stack the two cubes, UR5e, MuJoCo Warp), regenerated with visual observations
(seeds 1-34, image interval 5) and exported as LeRobot v2.1. Training data of the pi0.5 checkpoint ladder
(https://huggingface.co/adityx23/pi05-ocb-stack-ladder).

| | |
|---|---|
| Episodes | 3000 successful demos, 835,556 frames, 10 fps |
| Images | `observation.images.front`, `observation.images.wrist`: 128x128 RGB |
| State | `observation.state`: 19-d proprio |
| Action | `action`: 7-d absolute joint targets (6 arm + gripper) |
| Prompt | "stack the two cubes" |

`norm_stats/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json` holds the quantile norm stats openpi computed for the
ladder config. Use the dataset as `local/ocb_stack_3k` under `$HF_LEROBOT_HOME`.
"""


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(SRC, STAGE, copy_function=os.link, ignore=shutil.ignore_patterns(".cache"))
    (STAGE / NORM_DST).parent.mkdir(parents=True)
    shutil.copy(NORM, STAGE / NORM_DST)
    (STAGE / "README.md").write_text(README)
    files = [p for p in STAGE.rglob("*") if p.is_file()]
    n_src = sum(1 for p in SRC.rglob("*") if p.is_file() and ".cache" not in p.parts)
    assert len(files) == n_src + 2, (len(files), n_src)
    print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB in {STAGE}", flush=True)
    return len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="adityx23/ocb-stack-3k")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = stage()
    if a.dry_run:
        raise SystemExit(0)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(STAGE), num_workers=8)
    remote = [f for f in api.list_repo_files(a.repo, repo_type="dataset") if f != ".gitattributes"]
    print(f"uploaded: {len(remote)}/{n} files in https://huggingface.co/datasets/{a.repo}", flush=True)
    assert len(remote) == n, "file count mismatch: rerun to resume"
    shutil.rmtree(STAGE)
