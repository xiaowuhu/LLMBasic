import torch 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def clip(input, eps):
    return torch.clamp(input, 1-eps, 1+eps)

def min_ratio_a(ratio, Ai):
    y1 = ratio * Ai
    y2 = clip(ratio, 0.1) * Ai
    y = torch.min(y1, y2)
    return y

def plot(ax, ratio, y, label):
    ax.plot(ratio, y, label=label)
    ax.set_xlabel("$\pi / b$")
    ax.grid()
    ax.legend()

def main():
    ratio = torch.arange(0, 2, 0.01)
    Ai = -1
    y1 = min_ratio_a(ratio, Ai)
    Ai = 1
    y2 = min_ratio_a(ratio, Ai)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    plot(ax1, ratio, y1, "$A_b<0$")
    ax2 = fig.add_subplot(1,2,2)
    plot(ax2, ratio, y2, "$A_b>0$")
    plt.show()


main()
