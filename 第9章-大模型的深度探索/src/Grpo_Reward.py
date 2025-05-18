import numpy as np

score = np.array([0.0, 1.0, 2.0, 0.5])
std = np.std(score)
mean = np.mean(score)
A = (score - mean)/std
print("std=", std)
print("mean=", mean)
print("A=", A)
