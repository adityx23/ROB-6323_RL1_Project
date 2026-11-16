import pathlib, sys, time
t = time.time()
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
print("import", round(time.time() - t, 1), flush=True)
hard = sys.argv[1] == "hard"
b = benchmark.get_benchmark_dict()["libero_90"](); task = b.get_task(16); init = b.get_task_init_states(16)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
t = time.time(); env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128, hard_reset=hard)
print("hard" if hard else "soft", "make", round(time.time() - t, 1), flush=True)
for k in range(3):
    t = time.time(); env.reset(); env.set_init_state(init[k]); print("reset", round(time.time() - t, 1), flush=True)
    t = time.time()
    for i in range(100): o, _, _, _ = env.step([0] * 6 + [-1])
    print("100 steps", round(time.time() - t, 2), o["agentview_image"].shape, flush=True)
