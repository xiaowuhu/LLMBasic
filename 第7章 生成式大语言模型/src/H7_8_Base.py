import numpy as np

a = np.array([1.1, 1.6, 0.7])
print(a)
print("方差=",np.var(a),"均值=",np.mean(a))

b = a - 1.0
print(b)
print("方差=",np.var(b),"均值=",np.mean(b))
