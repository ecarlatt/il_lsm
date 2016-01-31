# -*- coding: utf-8 -*-
import numpy as np
import scipy as scip
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

# いつもの
# 1:学習時(input:d行が次元でn列のデータ)
def gaussker1(X,sigma):
    S = np.dot(X.T, X)
    size = np.shape(X)[1]
    s1 = np.zeros((size, size))
    s2 = np.zeros((size, size))
    for i in range(0, size):
        s1[i, :] = S[i, i]
        s2[:, i] = S[i, i]
    return scip.exp(-(s1 + s2 - 2 * S) / (2 * sigma ** 2))
# 2:出力時
def gaussker2(x,X,sigma):
    n1 = np.shape(x)[1]
    n2 = np.shape(X)[1]
    Sx = np.dot(x.T, x)
    SX = np.dot(X.T, X)
    s1 = np.zeros((n2, n1))
    s2 = np.zeros((n2, n1))
    for i in range(0, n2):
        s1[i, :] = SX[i, i]
    for i in range(0, n1):
        s2[:, i] = Sx[i, i]
    return scip.exp(-(s1 + s2 - 2 * np.dot(X.T, x)) / (2 * sigma ** 2))

# 線の出力
def expfig(x, xr, theta, sigma):
    KK = gaussker2(x, xr, sigma)
    t = np.dot(KK, theta)
    plt.plot(xr.T, t)

# とりあえず作ったけど、汎化誤差の推定量をどうするか...
# 訓練→誤差を返す
def Lasso_Huber_traintoer(x_train, x_test, y_train, y_test, xr, lam, sigma, eta):
    n = np.shape(x_train)[0]
    K = gaussker1(x_train.T, sigma)
    K2 = np.dot(K.T, K)
    Ky = np.dot(K.T, y_train)
    y_train_t = y_train
    theta0 = np.linalg.solve(K, y_train_t)
    for i in range(1, 1000):
        r = np.abs(np.dot(K, theta0) - y_train_t)
        w = np.ones(n)[np.newaxis].T
        w[r > eta] = eta / r[r > eta]
        W = np.diag(w[:, 0])
        theta = np.linalg.solve(np.dot(np.dot(K.T, W), K)
                           + lam * np.linalg.pinv(np.abs(np.diag(theta0[:,0])))
                           , np.dot(np.dot(K.T, W), y_train_t))
        if np.linalg.norm(theta0 - theta) < 0.0001:
            break
        theta0 = theta
    KK = gaussker2(x_train.T, xr, sigma)
    t = np.dot(KK, theta)
    a = np.zeros((np.shape(x_test)[0], 1), dtype=np.int)
    for i in range(0, np.shape(x_test)[0]):
        a[i] = np.argmin((xr.T - x_test[i]) * (xr.T - x_test[i]))
    y_test_s = t[a[:]][:, :, 0]
    return np.sum((y_test - y_test_s) ** 2)
# input:(sample_x, line_x, sample_y, lambda, bandwidth, eta, fold数), output:汎化誤差の推定量と回帰結果
def Lasso_Huber_crossval(x, xr, y, lam, sigma, eta, fold):
    n = np.shape(x)[1]
    kf = KFold(n, n_folds=fold, shuffle=True, random_state=1)
    i = 0
    error = np.zeros((fold, 1))
    for train, test in kf:
        x_train = x.T[train, :]
        y_train = y.T[train, :]
        x_test = x.T[test, :]
        y_test = y.T[test, :]
        error[i] = Lasso_Huber_traintoer(x_train, x_test, y_train, y_test, xr, lam, sigma, eta)
        i += 1
    print(np.mean(error))
    expfig(x, xr, theta, sigma)

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
    eta = 0.2
    """
    CV(?)するなら
    fold = 4
    Lasso_Huber_crossval(x, xr, y, lam, sigma, eta, fold)
    plt.scatter(x, y)
    plt.show()
    """
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
