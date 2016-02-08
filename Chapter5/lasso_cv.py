# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import scipy as scip
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

def gaussker1(X, sigma):
    X_sum = np.sum(X**2, 0)[np.newaxis]
    shape = np.size(X_sum)
    return scip.exp(-(np.matlib.repmat(X_sum.T, 1, shape) + np.matlib.repmat(X_sum, shape, 1)  -2 * np.dot(X.T, X))/(2 * sigma ** 2))
def gaussker2(x, X, sigma):
    return scip.exp(-(np.matlib.repmat((x ** 2), np.size(X), 1) + np.matlib.repmat((X ** 2).T, 1, np.size(x)) -2 * np.dot(X.T, x))/(2 * sigma ** 2))

# Lasso回帰のクロスバリデーション
# 訓練→誤差を返す
def lasso_traintoer(x_train, x_test, y_train, y_test, xr, lam, par):
    n = np.shape(x_train)[0]
    K = gaussker1(x_train.T, lam)
    K2 = np.dot(K.T, K)
    Ky = np.dot(K.T, y_train)
    theta0 = np.random.rand(n)[np.newaxis]
    theta0 = theta0.T
    for i in range(1, 1000):
        theta = np.linalg.solve(K2 + lam * np.linalg.pinv(np.abs(np.diag(theta0[:, 0]))), Ky)
        if np.linalg.norm(theta0 - theta) < 0.00001:
            break
        theta0 = theta
    KK = gaussker2(x_train.T, xr, lam)
    t = np.dot(KK, theta)
    a = np.zeros((np.shape(x_test)[0], 1), dtype=np.int)
    for i in range(0, np.shape(x_test)[0]):
        a[i] = np.argmin((xr.T - x_test[i]) * (xr.T - x_test[i]))
    y_test_s = t[a[:]][:, :, 0]
    return np.sum((y_test - y_test_s) ** 2)

# Lasso回帰のクロスバリデーション
# 訓練→誤差を返す
def Lasso_traintoer(x_train, x_test, y_train, y_test, xr, lam, sigma):
    n = np.shape(x_train)[0]
    K = gaussker1(x_train.T, sigma)
    K2 = np.dot(K.T, K)
    Ky = np.dot(K.T, y_train)
    theta0 = np.random.rand(n)[np.newaxis]
    theta0 = theta0.T
    for i in range(1, 1000):
        theta = np.linalg.solve(K2 + lam * np.linalg.pinv(np.abs(np.diag(theta0[:, 0]))), Ky)
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

# 線の出力
def Lasso_expfig(x, xr, y, lam, sigma):
    K = gaussker1(x, sigma)
    K2 = np.dot(K.T, K)
    Ky = np.dot(K.T, y.T)
    theta0 = np.random.rand(n)[np.newaxis]
    theta0 = theta0.T
    for i in range(1, 1000):
        theta = np.linalg.solve(K2 + lam * np.linalg.pinv(np.abs(np.diag(theta0[:,0]))), Ky)
        if np.linalg.norm(theta0 - theta) < 0.0001:
            break
        theta0 = theta
    KK = gaussker2(x, xr, sigma)
    t = np.dot(KK, theta)
    plt.plot(xr.T, t)

# Lasso K-Fold CV  input:(sample_x, line_x, sample_y, lambda, par, fold数), output:汎化誤差の推定量と回帰結果
def Lasso_crossval(x, xr, y, lam, sigma, fold):
    n = np.shape(x)[1]
    kf = KFold(n, n_folds=fold, shuffle=True, random_state=1)
    i = 0
    error = np.zeros((fold, 1))
    for train, test in kf:
        x_train = x.T[train, :]
        y_train = y.T[train, :]
        x_test = x.T[test, :]
        y_test = y.T[test, :]
        error[i] = Lasso_traintoer(x_train, x_test, y_train, y_test, xr, lam, sigma)
        i += 1
    print(np.mean(error))
    Lasso_expfig(x, xr, y, lam, sigma)

if __name__ == '__main__':
    n = 50
    N = 1000
    x = np.linspace(-3, 3, n)[np.newaxis]
    xr = np.linspace(-3, 3, N)[np.newaxis]
    y = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x + 0.4 * np.random.rand(50)[np.newaxis]
    # テスト
    sigma = 0.3
    lam = 0.2
    Fold = 4
    Lasso_crossval(x, xr, y, lam, sigma, Fold)
    plt.scatter(x, y)
    plt.xlim([-3, 3])
    plt.show()
