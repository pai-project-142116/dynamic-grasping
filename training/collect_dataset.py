"""軌道を画像10 Hzで観測し，予測器の学習データと平面校正を作る"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation import make_env, observe
from red_block_detection import detect, transform
from block_paths import KINDS, configure, path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/collected')
parser.add_argument('--limit', type=int, help='描画確認用の収集本数．省略時は126本')
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(31)
plan = []
for kind in KINDS:
    for split, count in [('train', 12), ('validation', 3), ('test', 3)]:
        for i in range(count):
            config = dict(id=len(plan), kind=kind, split=split, speed=[.05, .10, .20][i % 3],
                          center=[float(rng.uniform(.08, .12)), float(rng.uniform(.48, .52))],
                          radius=float(rng.uniform(.08, .10)), phase=float(rng.uniform(0, 2 * np.pi)),
                          delta=float(rng.uniform(-.4, .4)), direction=int(rng.choice([-1, 1])),
                          angle=float(rng.uniform(.4, 1.1)))
            plan.append(configure(config))
if (out / 'plan.json').exists():
    assert json.loads((out / 'plan.json').read_text()) == plan
else:
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
env = make_env()


def render(xy):
    # 学習時は腕を初期姿勢に固定する．画像入力と未来位置の教師を別々に保存する．
    env.physics.data.qpos[16:23] = [*xy, .0201, 1, 0, 0, 0]
    env.physics.data.qvel[16:22] = 0
    env.physics.forward()
    return observe(env)[1]


world = np.array([[-.04, .36], [.24, .36], [.24, .64], [-.04, .64]], dtype=np.float32)
pixels = np.array([detect(render(xy)) for xy in world], dtype=np.float32)
assert np.isfinite(pixels).all()
matrix = cv2.getPerspectiveTransform(pixels, world)
np.savez(out / 'calibration.npz', image_to_world=matrix, world=world, pixels=pixels)
times = np.arange(200) / 10
for config in plan[:args.limit]:
    file = out / f'episode{config["id"]:03d}.npz'
    if file.exists():
        continue
    truth, _ = path(config, times)
    observed, latencies = [], []
    for xy in truth:
        image = render(xy)
        before = time.perf_counter()
        observed.append(transform(detect(image), matrix))
        latencies.append((time.perf_counter() - before) * 1000)
    np.savez(file, times=times, observed=observed, truth=truth, detection_ms=latencies)
    error = np.linalg.norm(np.array(observed) - truth, axis=1) * 1000
    print(config['id'], config['kind'], config['split'], 'error_mm', np.nanmean(error), flush=True)
env.close()
