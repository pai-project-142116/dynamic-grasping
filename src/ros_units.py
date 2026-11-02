"""ROSでは角度をrad，片側グリッパー関節の変位をmで送受信する"""

import numpy as np
from gym_aloha.constants import normalize_puppet_gripper_position, unnormalize_puppet_gripper_position


def to_ros(values):
    values = np.array(values, dtype=float, copy=True)
    values[..., [6, 13]] = unnormalize_puppet_gripper_position(values[..., [6, 13]])
    return values


def to_policy(values):
    values = np.array(values, dtype=float, copy=True)
    values[..., [6, 13]] = normalize_puppet_gripper_position(values[..., [6, 13]])
    return values
