# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# データ生成
n = 10
N = 1000
x = np.linspace(-3, 3, n)[np.newaxis]
xr = np.linspace(-3, 3, N)[np.newaxis]
y = x + np.random.rand(10)
y[0, 9] = -10
y[0, 3] = -50
y[0, 4] = -25

phi = np.hstack([x.T, np.ones(n)[np.newaxis].T])
reg0 = np.dot(np.linalg.pinv(phi), y.T)
yt = y.T

"""
Tukeyと並べたかった
# Huber
eta = 1
for i in range(0, 1000):
    # 残差を求める
    r = np.abs(np.dot(phi, reg0) - yt)
    w = np.ones(n)[np.newaxis].T
    # 残差が大きければ重み付けを小さく取る(閾値はeta)
    w[r > eta] = eta / r[r > eta]
    W = np.diag(w[:, 0])
    # 重み付け最小二乗学習
    reg = np.linalg.solve(np.dot(np.dot(phi.T, W), phi), np.dot(np.dot(phi.T, W), yt))
    if np.linalg.norm(reg - reg0) < 0.0001:
        break
    reg0 = reg
plt.plot(xr.T, reg[0] * xr .T + reg[1], "g", label="Huber")
"""

# Tukey
eta = 10
for i in range(0, 1000):
    # 残差を求める
    r = np.abs(np.dot(phi, reg0) - yt)
    w = np.zeros(n)[np.newaxis].T
    w[r <= eta] = (1 - r[np.abs(r) <= eta] ** 2 / (eta ** 2)) ** 2
    W = np.diag(w[:, 0])
    # 重み付け最小二乗学習
    reg = np.linalg.solve(np.dot(np.dot(phi.T, W), phi), np.dot(np.dot(phi.T, W), yt))
    if np.linalg.norm(reg - reg0) < 0.0001:
        break
    reg0 = reg
plt.plot(xr.T, reg[0] * xr .T + reg[1], "r", label="Tukey")
plt.scatter(x, y)
plt.legend()
plt.ylim([-5, 5])
plt.show()
