"""未来の物体XYを教師として，軌道予測MLPをCPUで学習する"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/training',
                    help='prepare_dataset.pyの出力先．学習結果もここに保存')
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(27)
data = np.load(args.output / 'dataset.npz')
x, y = torch.from_numpy(data['train_x']), torch.from_numpy(data['train_y'])
vx, vy = torch.from_numpy(data['validation_x']), torch.from_numpy(data['validation_y'])
model = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
best, history = float('inf'), []
started = time.perf_counter()
for epoch in range(300):
    for batch in torch.randperm(len(x)).split(256):
        loss = ((model(x[batch]) - y[batch]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        validation = float(((model(vx) - vy) ** 2).mean())
    history.append(validation)
    # 検証誤差が最小の重みを採用する．テスト用21軌道は学習・選択に使わない．
    if validation < best:
        best, best_epoch = validation, epoch
        weights = {}
        for i, layer in enumerate([model[0], model[2], model[4]]):
            weights[f'w{i}'] = layer.weight.detach().numpy().copy()
            weights[f'b{i}'] = layer.bias.detach().numpy().copy()
        np.savez(args.output / 'trajectory_predictor.npz', **weights)
result = dict(seed=27, epochs=300, best_epoch=best_epoch, parameters=sum(p.numel() for p in model.parameters()),
              seconds=time.perf_counter() - started, device='cpu',
              validation_coordinate_rmse_mm=float(np.sqrt(best) * 50), history=history,
              torch_version=torch.__version__, numpy_version=np.__version__)
(args.output / 'training.json').write_text(json.dumps(result, indent=2) + '\n')
print({key: value for key, value in result.items() if key != 'history'})
