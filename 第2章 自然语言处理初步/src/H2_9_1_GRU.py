import math
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return 2.0 / (1.0 + np.exp(-2*x)) - 1.0


r_t = sigmoid(0.7 * 0.2 + (-0.3) * 0.5)
print("r_t = ", r_t)

u_t = sigmoid(0.7 * 0.3 + (-0.3) * 0.2)
print("u_t = ", u_t)

_h_t = tanh(0.7 * 0.4 +  r_t * (-0.3) * (-0.7))
print("_h_t = ", _h_t)

h_t = (1-u_t) * (-0.3) + u_t * _h_t
print("h_t = ", h_t)
