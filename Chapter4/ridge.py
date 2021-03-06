# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import scipy as scip
import matplotlib.pyplot as plt

def gaussker1(X, sigma):
    X_sum = np.sum(X**2, 0)[np.newaxis]
    shape = np.size(X_sum)
    return scip.exp(-(np.matlib.repmat(X_sum.T, 1, shape) + np.matlib.repmat(X_sum, shape, 1) -2 * np.dot(X.T, X))/(2 * sigma ** 2))
def gaussker2(x, X, sigma):
    return scip.exp(-(np.matlib.repmat((x ** 2), np.size(X), 1) + np.matlib.repmat((X ** 2).T, 1, np.size(x)) -2 * np.dot(X.T, x))/(2 * sigma ** 2))

if __name__ == '__main__':
    # データ生成
    n = 50
    N = 1000
    x = np.linspace(-3, 3, n)[np.newaxis]
    xr = np.linspace(-3, 3, N)[np.newaxis]
    y = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x + 0.4 * np.random.rand(50)
    # 回帰
    par = 0.5
    lam = 0.2
    K = gaussker1(x, par)
    theta = np.linalg.solve(np.dot(K, K) + lam * np.matrix(np.identity(n)), np.dot(K.T, y.T))
    KK = gaussker2(x, xr, par)
    t = np.dot(KK, theta)
    plt.scatter(x, y)
    plt.plot(xr.T, t)
    plt.xlim([-3, 3])
    plt.show()
