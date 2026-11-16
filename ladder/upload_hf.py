"""Upload the pi0.5 cube-stacking checkpoint ladder (weights only) to one public Hugging Face model repo.

Layout in the repo: <step>/params/... (orbax, EMA weights) and <step>/assets/... (norm stats), plus README.md.
Files are hard-linked into a staging dir (same filesystem, no copy) and sent with upload_large_folder, which is
resumable: rerunning continues where it stopped. Token: HF_TOKEN from ~/.config/ldr-hf/token (set by upload.sbatch).

    python upload_hf.py [--repo adityx23/pi05-ocb-stack-ladder] [--dry-run]
"""

import argparse
import os
import pathlib
import shutil

CK = pathlib.Path(os.environ["RLFT_ROOT"] + "/runs/ladder/checkpoints/pi05_ocbs3k/run")
STAGE = pathlib.Path(os.environ["RLFT_ROOT"] + "/runs/ladder/hf_stage")
STEPS = open(os.environ["PROJ"] + "/ladder/ladder_steps.txt").read().split()

README = """---
license: gemma
base_model: physical-intelligence/pi05_base
tags: [robotics, pi0.5, openpi, ocbench, manipulation]
---
# pi0.5 OCBench cube-stacking checkpoint ladder

pi0.5 (openpi / lair-openpi, JAX) fully fine-tuned from `pi05_base` on 3000 scripted demos of OCBench
`block-double-task2` (stack the two cubes, UR5e, MuJoCo Warp). Checkpoints at several training steps, meant as base
policies of different strength for RL fine-tuning.

| | |
|---|---|
| Data | 3000 successful demos, 835,556 frames, 10 fps; front + wrist camera 128x128; 19-d proprio state |
| Actions | 7-d absolute joint targets (6 arm + gripper); arm trained as deltas from chunk-start joints |
| Model | pi0.5, action horizon 30, train-time RTC (prefix delay ~ U[0, 15)), continuous state input |
| Training | batch 128, 30k steps, AdamW, peak lr 5e-5 cosine to 5e-6, 500 warmup, grad clip 1.0, EMA 0.999 |
| Steps | {steps} |

Each folder `<step>/` holds `params/` (EMA weights, orbax) and `assets/` (quantile norm stats).
Optimizer state is not included. Prompt: "stack the two cubes".
"""


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for s in STEPS:
        for sub in ("params", "assets"):
            src = CK / s / sub
            assert src.is_dir(), f"missing {src}"
            shutil.copytree(src, STAGE / s / sub, copy_function=os.link)
    (STAGE / "README.md").write_text(README.format(steps=", ".join(STEPS)))
    n = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {n} files, {size / 1e9:.1f} GB in {STAGE}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="adityx23/pi05-ocb-stack-ladder")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stage()
    if a.dry_run:
        raise SystemExit(0)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="model", folder_path=str(STAGE), num_workers=8)
    files = api.list_repo_files(a.repo)
    print(f"uploaded: {len(files)} files in https://huggingface.co/{a.repo}", flush=True)
    shutil.rmtree(STAGE)
