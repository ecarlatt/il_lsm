# utf-8 python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)
regdata = datasets.make_regression(100, 1, noise=5.0)
phi = np.c_[regdata[0], np.ones(100)]
reg = np.dot(np.linalg.pinv(phi), regdata[1])
print(reg)
xr = np.linspace(-3, 3, 1000)
plt.plot(xr, reg[0] * xr + reg[1])
plt.scatter(regdata[0], regdata[1])
plt.show()
