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


kl = []
c_len = []
reward = []
reward_std = []
loss = []

with open("../model/ch9/5/tldr.log", 'r', encoding='utf-8') as f:
    for line in f.readlines(): 
        data = ast.literal_eval(line)
        kl.append(data['kl'])
        c_len.append(data['completion_length'])
        reward.append(data['reward'])
        reward_std.append(data['reward_std'])
        loss.append(data['loss'])
        


show_curve(reward, "奖励函数值")
show_curve(kl, "KL散度")
show_curve(c_len, "输出长度")
show_curve(reward_std, "奖励函数值标准差")
show_curve(loss, "损失函数值")
