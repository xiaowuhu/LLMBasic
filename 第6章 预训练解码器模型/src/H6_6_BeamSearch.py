
import math

def score(arg, alpha):
    print("------")
    print(arg)
    L = len(arg)
    sum = 0
    for p in arg:
        sum += math.log(p)
    s = sum / math.pow(L, alpha)
    print(f"L = {L}, {sum:.4f} -> {s:.4f}")
    return s

candidate = [
    [0.4], [0.3], [0.4,0.4], [0.3,0.7], [0.3,0.7,0.45], [0.3,0.7,0.35]
]

alphas = [-0.1, 0, 0.75, 0.78, 1.0, 1.1]
for alpha in alphas:
    print(f"alpha={alpha}")
    for c in candidate:
        score(c, alpha)
