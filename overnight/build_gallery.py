"""Build the rollout gallery from eval results.

    python build_gallery.py EVAL_DIR OUT_DIR

Reads EVAL_DIR/results.jsonl, picks one success + one failure per (policy, suite, task), re-encodes them small
into OUT_DIR/v/, and writes OUT_DIR/index.html with the summary + per-task numbers inlined.
"""

import collections
import json
import math
import os
import pathlib
import subprocess
import sys

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
POLICY_NAMES = {"pi05": "π0.5", "act": "ACT"}
SUITE_NAMES = {"ocb_stack": "OCBench cube stacking"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ev, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    (out / "v").mkdir(parents=True, exist_ok=True)
    recs = {}
    for line in open(ev / "results.jsonl"):
        r = json.loads(line)
        recs[(r["policy"], r["suite"], r["task_id"], r["episode"])] = r  # last write wins on re-runs
    recs = list(recs.values())

    summ = collections.defaultdict(lambda: [0, 0])
    task = collections.defaultdict(lambda: [0, 0])
    names = {}
    for r in recs:
        summ[(r["policy"], r["suite"])][0] += r["success"]
        summ[(r["policy"], r["suite"])][1] += 1
        task[(r["policy"], r["suite"], r["task_id"])][0] += r["success"]
        task[(r["policy"], r["suite"], r["task_id"])][1] += 1
        names[(r["suite"], r["task_id"])] = r["task"]

    summary = []
    for (p, s), (k, n) in sorted(summ.items()):
        lo, hi = wilson(k, n)
        summary.append({"policy": p, "suite": s, "k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi})
    tasks = [{"policy": p, "suite": s, "task_id": t, "task": names[(s, t)], "k": k, "n": n}
             for (p, s, t), (k, n) in sorted(task.items())]

    n_ok, n_bad = int(os.environ.get("GALLERY_SUCC", 1)), int(os.environ.get("GALLERY_FAIL", 1))
    picks = collections.defaultdict(list)
    for r in sorted(recs, key=lambda r: r["episode"]):
        key = (r["policy"], r["suite"], r["task_id"], r["success"])
        if len(picks[key]) < (n_ok if r["success"] else n_bad) and (ev / r["video"]).exists():
            picks[key].append(r)
    videos = []
    for (p, s, t, ok), r in sorted(((k, r) for k, rs in picks.items() for r in rs), key=lambda kr: (kr[0], kr[1]["episode"])):
        name = f"{p}__{s}__t{t}__e{r['episode']}__{'s' if ok else 'f'}.mp4"
        dst = out / "v" / name
        if not dst.exists():
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(ev / r["video"]), "-vf", "scale=-2:192",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", str(dst)], check=True)
        videos.append({"policy": p, "suite": s, "task_id": t, "task": r["task"], "success": ok, "episode": r["episode"],
                       "steps": r["steps"], "src": f"v/{name}"})

    data = {"summary": summary, "tasks": tasks, "videos": videos, "policy_names": POLICY_NAMES,
            "suite_names": SUITE_NAMES}
    html = open(pathlib.Path(__file__).with_name("gallery_template.html")).read()
    (out / "index.html").write_text(html.replace("/*__DATA__*/null", json.dumps(data)))
    tot = sum(f.stat().st_size for f in (out / "v").iterdir())
    print(f"{len(recs)} episodes, {len(videos)} videos ({tot / 1e6:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
