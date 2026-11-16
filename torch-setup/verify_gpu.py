"""Sequential single-GPU checks with generated data and upstream smoke tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path(os.environ['RLFT_ROOT'])
job_id = os.environ['SLURM_JOB_ID']
root = repo / 'runs' / 'verification' / ('gpu-' + job_id)
root.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
env.update(RLFT_DATA=str(root / 'data'), RLFT_RUNS=str(root / 'runs'),
           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', SPIKE_NWORLD='8')
# Training's preemption handler otherwise calls scontrol requeue. Validation must
# stop on interruption; retain allocation identity in this parent and its logs.
env.pop('SLURM_JOB_ID', None)
stages = []

def sample_gpu(log):
    subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total',
                    '--format=csv'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=15)

def run(label, args, timeout):
    command = [sys.executable, *args]
    print('COMMAND', label, json.dumps(command), flush=True)
    started = time.monotonic()
    with (root / (label + '.log')).open('x') as log, (root / (label + '-gpu.csv')).open('x') as gpu_log:
        sample_gpu(gpu_log)
        with subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    sample_gpu(gpu_log)
                    if time.monotonic() - started > timeout:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise TimeoutError(label)
    stages.append({'stage': label, 'seconds': round(time.monotonic() - started, 1), 'exit_code': code})
    (root / 'stages.json').write_text(json.dumps(stages, indent=2))
    if code:
        print((root / (label + '.log')).read_text()[-16000:], flush=True)
        raise RuntimeError(f'{label} failed: {code}')

run('cuda', ['-c', 'import jax, torch; assert jax.default_backend() == "gpu"; '
             'assert torch.cuda.is_available(); print(jax.devices()); '
             'print(torch.cuda.get_device_name()); print(torch.ones(4, device="cuda").sum().item())'], 180)
run('simulation-render', ['scripts/env_spike.py'], 900)
spike_log = (root / 'simulation-render.log').read_text()
start = spike_log.rfind('\n{')
spike, _ = json.JSONDecoder().raw_decode(spike_log[start + 1:] if start >= 0 else spike_log)
assert spike['jax_matmul_ok'] and spike['torch_cuda']
assert spike['state_steps_per_s'] > 0 and spike['render_ms_per_world_3cams'] > 0
assert spike['image_stats']['mean'] > 0
run('generate-data', [os.environ['PROJ'] + '/torch-setup/generate_verification_data.py', env['RLFT_DATA']], 900)
common = ['-m', 'pytest', '-q', '-rA', '-o', 'cache_dir=' + str(root / 'pytest-cache')]
run('units', common + ['--basetemp=' + str(root / 'units-tmp'), 'tests', '--ignore=tests/smoke'], 1200)
# Exclude the deliberate preemption test: it asks SLURM to requeue the job.
run('base-workflow', common + ['--basetemp=' + str(root / 'base-tmp'),
    'tests/smoke/test_p0_smoke.py', '-k', 'not preemption'], 1200)
env['RLFT_SMOKE_METHODS'] = 'flow_bc,dsrl_na,dsrl_sac,expo,expo_ft,bestofn,residual,rlpd'
run('rl-workflows', common + ['--basetemp=' + str(root / 'rl-tmp'),
    'tests/smoke/test_rl_smoke.py'], 3600)
result = {'status': 'PASS', 'job_id': job_id, 'root': str(root), 'stages': stages,
          'scope': 'single-GPU simulation/render, generated state/visual data, units, base and all eight RL methods with resume',
          'limits': 'reduced generated data; preemption/requeue excluded; no learning-quality or released-dataset validation'}
(root / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
