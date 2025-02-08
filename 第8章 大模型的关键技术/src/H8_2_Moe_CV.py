import numpy as np

a = [0.8, 0.2, 0.1, 0.05]
s = np.std(a)
m = np.mean(a)
print(a)
print(f"方差{s}, 均值{m}, 方差/均值{s/m}")

a = [0.3, 0.3, 0.2, 0.2]
s = np.std(a)
m = np.mean(a)
print(a)
print(f"方差{s}, 均值{m}, 方差/均值{s/m}")

