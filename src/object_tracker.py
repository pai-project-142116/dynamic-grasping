"""10 Hzの画像から物体XYを検出し，過去の観測だけで未来位置を予測する"""

from collections import deque
from pathlib import Path

import numpy as np

from red_block_detection import detect, transform

FUTURE = np.arange(1, 6) / 10


class ObjectTracker:
    def __init__(self, method, assets):
        self.method = method
        self.matrix = np.load(Path(assets) / "camera_calibration.npz")["image_to_world"]
        self.weights = None
        if method == "mlp":
            with np.load(Path(assets) / "trajectory_predictor.npz") as saved:
                self.weights = dict(saved)
        self.points, self.times = deque(maxlen=10), deque(maxlen=10)
        self.latest = np.full(2, np.nan)
        self.prediction, self.last_seen = None, -np.inf

    def observe(self, image, stamp):
        # 固定カメラ・机上の赤いブロックを対象とする．真の物体座標は使わない．
        point = transform(detect(image), self.matrix)
        return self.update(point, stamp)

    def update(self, point, stamp):
        # この分離により，保存した検出結果でも予測器を検証できる
        self.latest = np.asarray(point).copy()
        self.points.append(self.latest.copy())
        self.times.append(stamp)
        if not np.isfinite(self.latest).all():
            return False
        self.last_seen = stamp
        self.prediction = np.tile(self.latest, (6, 1))
        points, times = np.array(self.points), np.array(self.times)
        if len(points) != 10 or not np.isfinite(points).all():
            return True
        if self.method == "velocity":
            coefficients = np.polynomial.polynomial.polyfit(times[-5:] - times[-1], points[-5:], 1)
            self.prediction[1:] = np.polynomial.polynomial.polyval(FUTURE, coefficients).T
        elif self.method == "mlp":
            # 相対XY・相対時刻・有効フラグの40値 → 0.1～0.5秒先のXY
            value = np.c_[(points - points[-1]) / .05, times - times[-1], np.ones(10)].ravel().astype(np.float32)
            for layer in range(3):
                value = self.weights[f"w{layer}"] @ value + self.weights[f"b{layer}"]
                if layer < 2:
                    value = np.maximum(value, 0)
            self.prediction[1:] = points[-1] + value.reshape(5, 2) * .05
        return True

    def predict(self, now):
        horizon = now - self.last_seen + .10
        return np.array([np.interp(horizon, np.r_[0, FUTURE], self.prediction[:, j]) for j in range(2)])
