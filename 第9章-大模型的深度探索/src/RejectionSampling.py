import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 提议分布：正态分布
def q_distribution(x, mu=0, sigma=1):
    #return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return norm.pdf(x, mu, sigma)

# 目标分布
def p_distribution(x):
    part1 = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    part2 = 1 + 0.5 * np.sin(5*x)
    return part1 * part2

def draw():
    fig = plt.figure(figsize=(8,4))

    ax1 = fig.add_subplot(1,2,1)
    x = np.linspace(-3, 3, 100)
    y = q_distribution(x, 0, 1)
    ax1.plot(x, y, linestyle='--', label="$q$")
    y = p_distribution(x)
    ax1.plot(x, y, linestyle='-', label="$p$")
    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    x = np.linspace(-3, 3, 100)
    y = q_distribution(x, 0, 1) * 1.5
    ax2.plot(x, y, linestyle='--', label="$q$")
    y = p_distribution(x)
    ax2.plot(x, y, linestyle='-', label="$p$")
    ax2.legend()

    plt.show()    

def draw_2(c, ax):
    x = np.linspace(-3, 3, 100)
    p = p_distribution(x)
    q = q_distribution(x)
    y = p / (c * q)
    ax.plot(x, y)
    ax.grid()



# 设置随机种子，保证结果可复现
#np.random.seed(42)
a, b = -3, 3 # 采样范围
c = 1.5  # 选择 M 使得 P(x) ≤ M * Q(x)

# 拒绝采样函数
def rejection_sampling(num_samples):
    samples = []
    while len(samples) < num_samples:
        # u = np.random.uniform(0, c*q)  # 生成随机数
        x = np.random.normal(loc=0, scale=1)
        q = q_distribution(x)
        u = np.random.uniform(0,1)  # 从 0,1 之间选择一个随机数
        #u = np.random.uniform(0, c*q)  # 生成随机数
        # u = np.random.uniform(a, b)  # 生成随机数
        p = p_distribution(x)
        if u < p / (c * q):
            samples.append(x)

    return np.array(samples)

# 生成 10000 个样本
samples = rejection_sampling(5000)

# 画出目标分布和采样分布
x_values = np.linspace(a, b, 1000)
y_values = p_distribution(x_values)

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(1,2,1)
draw_2(c, ax1)

ax2 = fig.add_subplot(1,2,2)
ax2.hist(samples, bins=50, density=True, alpha=0.5, label="拒绝采样结果")
ax2.plot(x_values, y_values, 'r-', label="目标分布")
ax2.legend()

plt.show()
