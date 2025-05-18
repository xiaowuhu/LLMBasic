import ast
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# # 移动平均函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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


loss = []
eval_loss = []
grad_norm = []
mean_token_accu = []
eval_mean_token_accu = []

line_no = 0
with open("../model/ch9/6/sft.log", 'r', encoding='utf-8') as f:
    for line in f.readlines(): 
        if line_no % 4 == 0:    # 4行数据是重复的，来自4块卡
            data = ast.literal_eval(line)
            if "loss" in data:  # 训练log
                grad_norm.append(data['grad_norm'])
                mean_token_accu.append(data['mean_token_accuracy'])
                loss.append(data['loss'])
            if "eval_loss" in data: # 验证 log
                for i in range(4):
                    eval_mean_token_accu.append(data['eval_mean_token_accuracy'])
                    eval_loss.append(data['eval_loss'])

show_curve2(loss, eval_loss, "训练集", "验证集", "损失函数")
show_curve2(mean_token_accu, eval_mean_token_accu, "训练集", "验证集", "准确率")

show_curve(grad_norm, "梯度范数")

