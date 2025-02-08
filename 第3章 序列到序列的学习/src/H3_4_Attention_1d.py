import math
import numpy as np

def gauss_dist(q, k):
    d = math.pow(q-k, 2)/2
    s = -d
    w = math.exp(-d)
    return d, s, w

def abs_dist(q, k):
    d = math.fabs(q-k)
    s = 1/d
    w = s
    return d, s, w

def compute(q, k, v, func):
    assert(len(k) == len(v))
    dists = []
    scores = []
    weights = []    
    total_weight = 0
    for i in range(len(k)):
        key = k[i]
        dist, score, weight = func(q, key)
        dists.append(dist)
        scores.append(score)
        weights.append(weight)
        total_weight += weight
    print("距离值:", " ".join("{:.4f}".format(i) for i in dists))
    print("分数值:", " ".join("{:.4f}".format(i) for i in scores))

    alphas = []
    for i in range(len(k)):
        w = weights[i] / total_weight
        alphas.append(w)
    print("权重值:", " ".join("{:.4f}".format(i) for i in alphas))

    z = 0
    for i in range(len(k)):
        z += alphas[i] * v[i]
    print("房价估计值: {:.2f}".format(z))

def Min_Max_Scalar(dict):
    k = []
    v = []
    for key, value in dict.items():
        k.append(key)
        v.append(value)
    x_min = np.min(k, axis=0)
    x_max = np.max(k, axis=0)
    x = (k - x_min) / (x_max - x_min)
    return x, x_min, x_max, v

if __name__=="__main__":
    # dict_price = {0.55:120, 0.62:135, 0.78:197, 0.97:211}
    # q = 0.66
    # dict_price = {5.5:120, 6.2:135, 7.8:197, 9.7:211}
    # q = 6.6
    dict_price = {55:120, 62:135, 78:197, 97:211}
    q = 66

    k, k_min, k_max, v = Min_Max_Scalar(dict_price)
    print("k=", k)
    print("v=", v)
    q = (q - k_min) / (k_max - k_min)
    print("q=", q)

    print("--- 高斯距离 ----")
    compute(q, k, v, gauss_dist)
    print("---- 曼哈顿距离 ----")
    compute(q, k, v, abs_dist)
