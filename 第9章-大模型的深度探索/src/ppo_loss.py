import numpy as np

kl = np.array([0.5, 0.8, 2.3, 2.6, 1.9, 0.4, 2.1, 0.6, 1.6, 1.8])
rm = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0., 1.8])
score = rm - 0.2 * kl
print(score)
