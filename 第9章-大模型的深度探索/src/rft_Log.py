import ast
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


import numpy as np
import matplotlib.pyplot as plt

# # 示例数据
# x = np.linspace(0, 10, 100)
# y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# # 移动平均函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# # 平滑数据
# y_smooth = moving_average(y, window_size)

# # 绘图
# plt.plot(x, y, label='Original')
# plt.plot(x[:len(y_smooth)], y_smooth, label='Smoothed (Moving Average)', color='red')
# plt.legend()
# plt.show()


def show_curve(y, title):
    window_size = 100
    y_smooth = moving_average(y, window_size)
    plt.plot(y_smooth)
    plt.title(title)
    plt.grid()
    plt.show()

def show_curve2(y1, y2, l1, l2, title):
    window_size = 100
    y1_smooth = moving_average(y1, window_size)
    y2_smooth = moving_average(y2, window_size)
    plt.plot(y1_smooth, linestyle='-', label=l1)
    plt.plot(y2_smooth, linestyle=':', label=l2)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

kl = []
c_len = []
reward = []
reward_std = []
loss = []
grad_norm = []

line_no = 0
with open("../model/ch9/7/rft.log", 'r', encoding='utf-8') as f:
    for line in f.readlines(): 
        if line_no % 4 == 0:    # 4行数据是重复的，来自4块卡
            data = ast.literal_eval(line)
            kl.append(data['kl'])
            c_len.append(data['completion_length'])
            reward.append(data['reward'])
            reward_std.append(data['reward_std'])
            loss.append(data['loss'])
            grad_norm.append(data['grad_norm'])
        
#show_curve2(kl, loss, "KL散度", "损失函数值", "KL散度和损失函数值")
#show_curve2(reward, reward_std, "奖励函数值", "奖励函数值标准差", "奖励函数值和标准差")

show_curve(reward, "奖励函数值")
show_curve(kl, "KL散度")
show_curve(loss, "损失函数值")
show_curve(c_len, "输出长度")
show_curve(reward_std, "奖励函数值标准差")
