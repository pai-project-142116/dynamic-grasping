"""画像予測による手先XY補正と，左右腕の連続した関節指令を計算する"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from object_tracker import ObjectTracker

SITE = "cali_right_site1"
JOINTS = [f"vx300s_right/{name}" for name in
          ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]]
ARMS = np.r_[0:6, 7:13]


class Correction:
    def __init__(self, env, method="mlp", assets=None):
        self.method = method
        self.tracker = ObjectTracker(method, assets or Path(__file__).resolve().parents[1] / "assets")
        self.physics = env.physics.copy(share_model=True)
        self.limits = self.physics.named.model.jnt_range[JOINTS].copy()
        self.command = env.task.get_qpos(env.physics).copy()
        self.velocity = np.zeros(12)
        self.reference = None
        self.offset, self.applied_offset = np.zeros(2), np.zeros(2)
        self.approached, self.close_count, self.close_time = False, 0, None

    def set_reference(self, reference):
        # 新しいchunkの観測位置へ，残っている補正値の基準も移す
        if self.method != "none" and reference is not None and self.reference is not None and self.close_time is None:
            self.offset += self.reference - reference
        self.reference = reference

    def apply(self, action, state, now):
        physics, tracker = self.physics, self.tracker
        physics.data.qpos[8:14] = state[7:13]
        physics.forward()
        actual = physics.named.data.site_xpos[SITE].copy()
        if actual[2] < .05 and state[13] > .6:
            self.approached = True
        closing = (self.approached and action[13] < .45 and state[13] < .8 and actual[2] < .08
                   and np.linalg.norm(actual[:2] - tracker.latest) < .04)
        self.close_count = self.close_count + 1 if closing else 0
        if self.close_count >= 3 and self.close_time is None:
            self.close_time = now
            self.offset = self.applied_offset.copy()

        offset, predicted = np.zeros(2), np.full(2, np.nan)
        if self.close_time is not None:
            offset = self.offset if now - self.close_time < .2 else self.offset * np.exp(-.02 / .3)
        elif self.method != "none" and self.reference is not None and tracker.prediction is not None:
            if now - tracker.last_seen <= .3:
                predicted = tracker.predict(now)
                offset = predicted - self.reference
            else:
                offset = self.offset * np.exp(-.02 / .1)
        norm = np.linalg.norm(offset)
        if norm > .15:
            offset *= .15 / norm
        if self.close_time is None and now - tracker.last_seen <= .3:
            offset = self.offset + (1 - np.exp(-.02 / .06)) * (offset - self.offset)
        self.offset = offset

        output = action.copy()
        if self.method != "none" and np.linalg.norm(offset) > 1e-6:
            physics.data.qpos[8:14] = action[7:13]
            physics.forward()
            nominal = physics.named.data.site_xpos[SITE].copy()
            rotation = physics.named.data.site_xmat[SITE].reshape(3, 3).copy()
            target = nominal.copy()
            target[:2] += offset
            blend = np.clip((.20 - actual[2]) / .10, 0., 1.)
            if self.close_time is None and np.isfinite(predicted).all():
                # 接近後は予測物体位置へ合わせる．1 cmは把持点とEE原点の差の近似．
                aligned = predicted - .01 * rotation[:2, 2]
                target[:2] = (1 - blend) * target[:2] + blend * aligned
            target[:2] = nominal[:2] + blend * (target[:2] - nominal[:2])
            self.applied_offset = target[:2] - nominal[:2]
            output[7:13] = self.solve_position(target)

        # 左腕も含めて，速度2 rad/s・加速度12 rad/s²以内の指令へ接続する
        desired = np.clip((output[ARMS] - self.command[ARMS]) / .02, -2., 2.)
        self.velocity += np.clip(desired - self.velocity, -.24, .24)
        self.command[ARMS] += self.velocity * .02
        output[ARMS] = self.command[ARMS]
        self.command = output.copy()
        return output

    def solve_position(self, target):
        # 位置と「左右の指が水平」の1条件を，小さな関節変化で満たす
        physics, previous = self.physics, self.command[7:13]
        physics.data.qpos[8:14] = previous
        physics.forward()
        jacobian = np.zeros((3, physics.model.nv))
        rotation_jacobian = np.zeros_like(jacobian)
        mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, jacobian, rotation_jacobian,
                         physics.model.name2id(SITE, "site"))
        closing_axis = physics.named.data.site_xmat[SITE].reshape(3, 3)[:, 1]
        level_jacobian = np.cross(closing_axis, [0., 0., 1.]) @ rotation_jacobian[:, 8:14]
        matrix = np.vstack([jacobian[:, 8:14], .2 * level_jacobian, .02 * np.eye(6)])
        error = np.r_[target - physics.named.data.site_xpos[SITE], -.2 * closing_axis[2], np.zeros(6)]
        lower = np.maximum(self.limits[:, 0] - previous, -.04)
        upper = np.minimum(self.limits[:, 1] - previous, .04)
        solution = lsq_linear(matrix, error, bounds=(lower, upper), method="bvls", tol=1e-6)
        return previous + solution.x

    def close(self):
        self.physics.free()
