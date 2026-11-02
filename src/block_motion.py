"""MuJoCo内のブロックを動かし，同じ腕の両指接触が続いたら軌道拘束を解除する"""

import numpy as np

from block_paths import path


class BlockMotion:
    def __init__(self, config):
        self.config = config
        self.count = np.zeros(2, dtype=int)
        self.released, self.release_time = False, None

    def move(self, env):
        if self.released or not self.config['speed']:
            return
        xy, velocity = path(self.config, env.physics.data.time)
        env.physics.data.qpos[16:23] = [*xy, .0201, 1, 0, 0, 0]
        env.physics.data.qvel[16:22] = [*velocity, 0, 0, 0, 0]
        env.physics.forward()

    def check_grasp(self, contacts, time):
        # 左右の腕を別々に数える．3制御ステップ接触した後は物理に任せる．
        self.count = np.where(contacts, self.count + 1, 0)
        if not self.released and self.count.max() >= 3:
            self.released, self.release_time = True, time
