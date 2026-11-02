"""画像中の赤いブロックの重心を検出し，校正済みの平面座標へ変換する"""

import cv2
import numpy as np


def detect(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 100, 60), (179, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or max(map(cv2.contourArea, contours)) < 12:
        return np.array([np.nan, np.nan])
    moment = cv2.moments(max(contours, key=cv2.contourArea))
    return np.array([moment['m10'] / moment['m00'], moment['m01'] / moment['m00']])


def transform(points, matrix):
    points = np.asarray(points)
    homogeneous = np.c_[points.reshape(-1, 2), np.ones(points.size // 2)] @ matrix.T
    return (homogeneous[:, :2] / homogeneous[:, 2:]).reshape(points.shape)
