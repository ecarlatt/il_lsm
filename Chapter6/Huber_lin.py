# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# データ生成
n = 10
N = 1000
x = np.linspace(-3, 3, n)[np.newaxis]
xr = np.linspace(-3, 3, N)[np.newaxis]
y = x + np.random.rand(10)
y[0, 9] = -3

phi = np.hstack([x.T, np.ones(n)[np.newaxis].T])
# 通常の最小二乗学習の結果を初期値に採用
reg0 = np.dot(np.linalg.pinv(phi), y.T)
eta = 1
yt = y.T
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
plt.plot(xr.T, reg[0] * xr .T + reg[1])
plt.scatter(x, y)
plt.show()
