"""Does the eval's execution path reproduce a demo? Replay successful OCBench stacking demos from their recorded
initial qpos/qvel two ways and report success:
  exact : every control step's own target (relabelled abs target), i.e. the demo itself (sanity check)
  hold5 : the stored LeRobot row target held for 5 control steps (what ocb_eval.py executes)
"""

import glob

import numpy as np

from rlft_bench.actions.spaces import JointSpace, relabel_episode_abs
from rlft_bench.data.ocb_import import load_ocb
from rlft_bench.envs.ocbench_adapter import OCBenchAdapter

TASK = "block-double-task2"
N = 6
paths = sorted(p for p in glob.glob(f"/scratch/aap9348/rlft-bench/data/ocbench/regen/visual-{TASK}-v0/i5/train-p000.npz"))
ep = load_ocb(paths, TASK, max_episodes=N + 4, successful_only=True)
raw = np.load(paths[0], mmap_mode="r")
term = np.where(raw["terminals"])[0]
starts = np.concatenate([[0], term[:-1] + 1])

env = OCBenchAdapter(TASK, num_envs=N, visual=False, seed=1)
js = JointSpace.from_adapter(env)
nq = env.env.data.qpos.numpy().shape[1]


def restore(e_ids):
    env.reset(seeds=np.arange(1, N + 1, dtype=np.uint32))
    st = env.get_sim_state()
    for w, e in enumerate(e_ids):
        s0 = starts[e]
        st.arrays["data.qpos"][w] = raw["qpos"][s0][:nq]
        st.arrays["data.qvel"][w] = raw["qvel"][s0][: st.arrays["data.qvel"].shape[1]]
    env.set_sim_state(st, np.arange(N))


# successful episodes in this part file (load_ocb keeps file order, so take the first N successful)
import pickle  # noqa: E402

meta = pickle.load(open(paths[0].replace(".npz", "-metadata.pkl"), "rb"))
succ_ids = [i for i, m in enumerate(meta["episodes"]) if m["success"]][:N]
eps = [ep.episode(k) for k in range(N)]
for mode in ("exact", "hold5"):
    restore(succ_ids)
    T = max(int(d["length"]) for d in eps)
    acts = [relabel_episode_abs(js, d["actions_rel"], d["q"][:-1])[0] for d in eps]
    success = np.zeros(N, bool)
    target = np.zeros((N, 7), np.float32)
    for t in range(T + 200):
        for w, d in enumerate(eps):
            L = int(d["length"])
            if mode == "exact":
                target[w] = acts[w][min(t, L - 1)]
            else:  # the row stored at the last image step <= t
                r = min((t // 5) * 5, L - 1)
                target[w] = acts[w][r]
        _, _, term_, trunc, info = env.step(target, absolute=True)
        success |= np.asarray(info.get("success", np.zeros(N, bool)), bool)
    print(f"{mode}: success {success.sum()}/{N}  {success.astype(int)}", flush=True)
