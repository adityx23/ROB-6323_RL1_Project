"""Small CPU workflow verification; run only in the approved batch allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(os.environ['RLFT_ROOT'])
root = repo / 'runs' / 'verification' / ('cpu-' + os.environ['SLURM_JOB_ID'])
root.mkdir(parents=True, exist_ok=False)

def run(label, args):
    command = [sys.executable, *args]
    print('COMMAND', label, json.dumps(command), flush=True)
    with (root / (label + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        print((root / (label + '.log')).read_text()[-12000:], flush=True)
        raise RuntimeError(f'{label} failed: {result.returncode}')

reuse_base = os.environ.get('RLFT_REUSE_BASE')
if not reuse_base:
    run('units', ['-m', 'pytest', '-q', '-rA', '-o', 'cache_dir=' + str(root / 'pytest-cache'),
              '--basetemp=' + str(root / 'pytest-tmp'),
              'tests/test_actions.py', 'tests/test_p1_units.py', 'tests/test_p2_units.py',
              'tests/test_envs.py::test_toy_snapshot_restore_exact',
              'tests/test_p3.py::test_value_video_composition'])
base = Path(reuse_base) if reuse_base else root / 'base'
base_args = ['scripts/train_base.py', 'env=toy_pointmass', 'obs=state', 'seed=0',
             'name=torch_cpu_verification', f'run_dir={base}', 'data.num_demos=5',
             'env.eval_envs=2', 'action.H=5', 'action.C=5', 'base.hidden=32', 'base.depth=2',
             'base.feat_dim=16', 'base.num_steps=2', 'train.batch=8', 'train.log_every=2',
             'train.ckpt_steps=[10]', 'train.eval_every=20', 'eval.num_videos=0',
             'eval.video_candidates=0', 'wandb.mode=disabled']
if not reuse_base:
    run('base', base_args + ['train.steps=20'])
    run('base-resume', base_args + ['train.steps=24'])
    assert 'resumed from' in (root / 'base-resume.log').read_text()
else:
    print('Reusing verified base', str(base), flush=True)
    assert 'resumed from' in (base.parent / 'base-resume.log').read_text()
assert (base / 'policy' / 'step_24').exists()
assert (base / 'policy' / 'norm_stats.json').is_file()
rl = root / 'rl'
rl_args = ['scripts/train_rl.py', f'base.run={base}', 'env=toy_pointmass', 'obs=state',
           'method=dsrl_na', 'seed=0', 'name=torch_cpu_verification', f'run_dir={rl}',
           'env.num_envs=2', 'env.eval_envs=2', 'action.C=5', 'data.num_demos=2',
           'critic.hidden=[32,32]', 'critic.feat_dim=16', 'method.actor_hidden=[32,32]',
           'train.batch=8', 'train.warmup_transitions=1', 'train.utd=0.5',
           'train.ckpt_every=2', 'train.eval_every=80', 'replay.capacity=512',
           'replay.demo_capacity=256', 'eval.num_videos=0', 'eval.video_candidates=0',
           'eval.record_decisions=0', 'wandb.mode=disabled']
for key in ('sensitivity', 'shuffle', 'grads', 'td_health', 'net_health', 'actor', 'bestofn', 'landscape'):
    rl_args.append(f'introspect.{key}.enabled=false')
run('rl', rl_args + ['train.env_steps=80'])
run('rl-resume', rl_args + ['train.env_steps=100'])
assert 'resumed from' in (root / 'rl-resume.log').read_text()
for path in (base, rl):
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    assert any('eval/success_rate' in row for row in rows), str(path)
    assert list((path / 'ckpt').glob('step_*')), str(path)
    assert (path / 'config.yaml').is_file()
result = {'status': 'PASS', 'scope': 'CPU units, toy base/RL/eval/checkpoint/resume',
          'root': str(root), 'seed': 0, 'gpu_tested': False, 'ocbench_tested': False}
(root / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
