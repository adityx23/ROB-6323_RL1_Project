"""Generate small local OCBench oracle datasets for workflow tests; no downloads."""
from pathlib import Path
import pickle
import sys

sys.path.insert(0, '/scratch/aap9348/rlft-bench')
from rlft_bench.data.regenerate import GenKnobs, generate

root = Path(sys.argv[1])
for visual, count, relative in (
    (False, 16, 'full/block-single-task1-v0/000.npz'),
    (True, 8, 'visual/visual-block-single-task1-v0/000.npz'),
):
    output = root / relative
    assert not output.exists(), output
    result = generate('block-single-task1', visual, count, str(output),
                      GenKnobs(observation_interval=5), seed=0, max_parallel=2, num_val=4)
    print(result, flush=True)
    for path in (output, output.with_name('000-val.npz')):
        assert path.is_file() and path.stat().st_size > 0, path
        metadata = path.with_name(path.stem + '-metadata.pkl')
        with metadata.open('rb') as stream:
            episodes = pickle.load(stream)['episodes']
        successes = sum(bool(episode['success']) for episode in episodes)
        print(f'{path}: episodes={len(episodes)} successes={successes} bytes={path.stat().st_size}', flush=True)
        assert successes > 0, f'No successful demonstrations in {path}; stop rather than silently retry'
