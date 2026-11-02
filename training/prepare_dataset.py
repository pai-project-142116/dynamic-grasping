"""別エピソードの訓練・検証データから，過去10点と未来5点の組を作る"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=ROOT / 'data/trajectories')
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/training')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plan = json.loads((args.data / 'plan.json').read_text())
arrays, manifest = {}, {}
for split in ['train', 'validation']:
    x, y, ids = [], [], []
    for config in plan:
        if config['split'] != split:
            continue
        ids.append(config['id'])
        with np.load(args.data / f'episode{config["id"]:03d}.npz') as data:
            for i in range(9, len(data['times']) - 5):
                points, times = data['observed'][i-9:i+1], data['times'][i-9:i+1]
                # 欠測を真値で補わない．推論時と同じ相対座標・時刻で入力する．
                if not np.isfinite(points).all():
                    continue
                feature = np.c_[(points - points[-1]) / .05, times - times[-1], np.ones(10)]
                x.append(feature.ravel().astype(np.float32))
                y.append(((data['truth'][i+1:i+6] - data['observed'][i]) / .05).ravel())
    arrays[split + '_x'] = np.array(x, dtype=np.float32)
    arrays[split + '_y'] = np.array(y, dtype=np.float32)
    manifest[split] = dict(episodes=ids, windows=len(x))
np.savez(args.output / 'dataset.npz', **arrays)
(args.output / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
print({key: value['windows'] for key, value in manifest.items()})
