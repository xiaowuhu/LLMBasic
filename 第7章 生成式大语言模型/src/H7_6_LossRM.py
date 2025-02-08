import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

x = np.linspace(-5,5, 100)
y = sigmoid(x)
plt.plot(x, y, ":", label="sigmoid")

z = -np.log(y)
plt.plot(x, z, "-", label="-log")
plt.grid()
plt.ylabel("loss")
plt.xlabel("$r_w - r_l$")
plt.legend()
plt.show()

