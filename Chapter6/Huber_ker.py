# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import scipy as scip
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

def gaussker1(X, sigma):
    X_sum = np.sum(X**2, 0)[np.newaxis]
    shape = np.size(X_sum)
    return scip.exp(-(np.matlib.repmat(X_sum.T, 1, shape) + np.matlib.repmat(X_sum, shape, 1) - 2 * np.dot(X.T, X))/(2 * sigma ** 2))
def gaussker2(x, X, sigma):
    return scip.exp(-(np.matlib.repmat((x ** 2), np.size(X), 1) + np.matlib.repmat((X ** 2).T, 1, np.size(x)) -2 * np.dot(X.T, x))/(2 * sigma ** 2))

# 線の出力
def expfig(x, xr, theta, sigma):
    KK = gaussker2(x, xr, sigma)
    t = np.dot(KK, theta)
    plt.plot(xr.T, t)

if __name__ == '__main__':
    # サンプルデータ
    n = 50
    N = 1000
    x = np.linspace(-3, 3, n)[np.newaxis]
    xr = np.linspace(-3, 3, N)[np.newaxis]
    y = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x + 0.4 * np.random.rand(50)[np.newaxis]
    y[:, 25] = -1

    sigma = 0.3
    lam = 0.1
    eta = 0.1

    yt = y.T
    K = gaussker1(x, sigma)
    K2 = np.dot(K.T, K)
    Ky = np.dot(K.T, y.T)
    # 制約なしバージョンを初期値に採用
    theta0 = np.linalg.solve(K, y.T)
    for i in range(1, 1000):
        r = np.abs(np.dot(K, theta0) - yt)
        w = np.ones(n)[np.newaxis].T
        w[r > eta] = eta / r[r > eta]
        W = np.diag(w[:, 0])
        theta = np.linalg.solve(np.dot(np.dot(K.T, W), K)
                               + lam * np.linalg.pinv(np.abs(np.diag(theta0[:,0])))
                               , np.dot(np.dot(K.T, W), yt))
        if np.linalg.norm(theta0 - theta) < 0.00001:
            break
        theta0 = theta
    expfig(x, xr, theta, sigma)
    plt.scatter(x, y)
    plt.show()
