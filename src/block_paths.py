"""7種類の軌道についてXY座標と速度を計算する"""

import numpy as np

KINDS = ['x_line', 'y_line', 'diagonal', 'circle', 'square', 'lissajous12', 'lissajous34']


def shape(config, phase):
    p = np.asarray(phase)
    kind = config['kind']
    if kind in ['x_line', 'y_line', 'diagonal']:
        angle = {'x_line': 0., 'y_line': np.pi / 2, 'diagonal': config.get('angle', np.pi / 4)}[kind]
        direction = np.array([np.cos(angle), np.sin(angle)])
        return (config['radius'] * np.sin(p)[..., None] * direction,
                config['radius'] * np.cos(p)[..., None] * direction)
    if kind == 'circle':
        xy = np.stack([np.cos(p), np.sin(p)], axis=-1)
        derivative = np.stack([-np.sin(p), np.cos(p)], axis=-1)
    elif kind == 'square':
        vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        progress = (p / (2 * np.pi) % 1) * 4
        edge = np.floor(progress).astype(int)
        side = vertices[edge + 1] - vertices[edge]
        xy = vertices[edge] + side * (progress - edge)[..., None]
        derivative = side * 2 / np.pi
    else:
        a, b = (1, 2) if kind == 'lissajous12' else (3, 4)
        delta = config.get('delta', 0.)
        xy = np.stack([np.sin(a * p), np.sin(b * p + delta)], axis=-1)
        derivative = np.stack([a * np.cos(a * p), b * np.cos(b * p + delta)], axis=-1)
    return config['radius'] * xy, config['radius'] * derivative


def configure(config):
    # 曲線の形が違っても，指定した最大XY速度に揃える
    config = dict(config)
    _, derivative = shape(config, np.linspace(0, 2 * np.pi, 20001))
    config['omega'] = config['speed'] / np.linalg.norm(derivative, axis=-1).max()
    config['period'] = 2 * np.pi / config['omega'] if config['omega'] else 0.
    return config


def path(config, time):
    if config['speed'] == 0:
        return np.broadcast_to(config['center'], np.shape(time) + (2,)).copy(), np.zeros(np.shape(time) + (2,))
    omega = config['omega'] * config.get('direction', 1)
    xy, derivative = shape(config, omega * np.asarray(time) + config['phase'])
    return xy + config['center'], derivative * omega
