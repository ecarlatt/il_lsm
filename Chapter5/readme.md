
## Chap5.スパース学習


```python
import numpy as np
import numpy.matlib
import scipy as scip
import matplotlib.pyplot as plt
%matplotlib inline
def gaussker1(X, sigma):
    X_sum = np.sum(X**2, 0)[np.newaxis]
    shape = np.size(X_sum)
    return scip.exp(-(np.matlib.repmat(X_sum.T, 1, shape) + np.matlib.repmat(X_sum, shape, 1) -2 * np.dot(X.T, X))/(2 * sigma ** 2))
def gaussker2(x, X, sigma):
    return scip.exp(-(np.matlib.repmat((x ** 2), np.size(X), 1) + np.matlib.repmat((X ** 2).T, 1, np.size(x)) -2 * np.dot(X.T, x))/(2 * sigma ** 2))
```

### Lasso


```python
#Lasso
from numpy.random import *
n = 50
N = 1000
x = np.linspace(-3,3,n)[np.newaxis]
xr = np.linspace(-3,3,N)[np.newaxis]
y = np.sin(np.pi*x)/(np.pi*x) + 0.1*x + 0.4*rand(50)[np.newaxis]
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x2958dd4ba20>




![png](output_3_1.png)



```python
sigma = 0.3
lam = 0.3
K = gaussker1(x, sigma)
K2 = np.dot(K.T,K)
Ky = np.dot(K.T,y.T)
theta0 = np.random.rand(n)[np.newaxis]
theta0 = theta0.T
for i in range(1,1000):
    theta =np.linalg.solve(K2 + lam*np.linalg.pinv(np.abs(np.diag(theta0[:,0]))), Ky)
    if np.linalg.norm(theta0-theta) < 0.00001:
        break
    theta0 = theta
KK = gaussker2(x, xr, sigma)
t = np.dot(KK, theta)
plt.scatter(x, y)
plt.plot(xr.T, t)
plt.xlim([-3, 3])
```




    (-3, 3)




![png](output_4_1.png)



```python
print(theta[theta > 0.0001])
```

    [  8.53020878e-02   5.20415448e-04   3.29067994e-01   5.49472655e-02
       5.76577389e-02   6.74889297e-01   1.43628575e-02   1.88140593e-01
       7.22102049e-01   1.64639606e-02   2.47892719e-02   5.35450344e-02
       2.25115658e-02   5.31954368e-03   3.10864509e-01   1.28745547e-01
       5.44243864e-02   3.00042113e-01]
    

### Lasso CV


```python
#Lasso回帰のクロスバリデーション
from sklearn.cross_validation import train_test_split as crv
from sklearn.cross_validation import KFold
# 訓練→誤差を返す
def Lasso_traintoer(x_train, x_test, y_train, y_test, xr, lam, sigma):
    n = np.shape(x_train)[0]
    K = gaussker1(x_train.T, sigma)
    K2 = np.dot(K.T,K)
    Ky = np.dot(K.T, y_train)
    theta0 = np.random.rand(n)[np.newaxis]
    theta0 = theta0.T
    for i in range(1,1000):
        theta =np.linalg.solve(K2 + lam*np.linalg.pinv(np.abs(np.diag(theta0[:,0]))), Ky)
        if np.linalg.norm(theta0-theta) < 0.0001:
            break
        theta0 = theta
    KK = gaussker2(x_train.T, xr, sigma)
    t = np.dot(KK, theta)
    a = np.zeros((np.shape(x_test)[0], 1), dtype=np.int)
    for i in range(0, np.shape(x_test)[0]):
        a[i] = np.argmin((xr.T - x_test[i]) * (xr.T - x_test[i]))
    y_test_s = t[a[:]][:, :, 0]
    return np.sum((y_test - y_test_s) ** 2)
#線の出力
def Lasso_expfig(x, xr, y, lam, sigma):
    K = gaussker1(x, sigma)
    K2 = np.dot(K.T,K)
    Ky = np.dot(K.T,y.T)
    theta0 = np.random.rand(n)[np.newaxis]
    theta0 = theta0.T
    for i in range(1,1000):
        theta =np.linalg.solve(K2 + lam*np.linalg.pinv(np.abs(np.diag(theta0[:,0]))), Ky)
        if np.linalg.norm(theta0-theta) < 0.0001:
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
```


```python
Fold = 4
```


```python
sigma = 0.3
lam = 0.3
Lasso_crossval(x, xr, y, lam , sigma, Fold)
plt.scatter(x, y)
plt.xlim([-3, 3])
```

    0.468384561304
    




    (-3, 3)




![png](output_9_2.png)



```python
sigma = 0.2
lam = 0.3
Lasso_crossval(x, xr, y, lam , sigma, Fold)
plt.scatter(x, y)
plt.xlim([-3, 3])
```

    0.669178034062
    




    (-3, 3)




![png](output_10_2.png)



```python
sigma = 0.4
lam = 0.3
Lasso_crossval(x, xr, y, lam , sigma, Fold)
plt.scatter(x, y)
plt.xlim([-3, 3])
```

    0.356587970794
    




    (-3, 3)




![png](output_11_2.png)



```python
sigma = 0.3
lam = 0.4
Lasso_crossval(x, xr, y, lam , sigma, Fold)
plt.scatter(x, y)
plt.xlim([-3, 3])
```

    0.579448503492
    




    (-3, 3)




![png](output_12_2.png)



```python
sigma = 0.3
lam = 0.2
Lasso_crossval(x, xr, y, lam , sigma, Fold)
plt.scatter(x, y)
plt.xlim([-3, 3])
```

    0.389371565578
    




    (-3, 3)




![png](output_13_2.png)

