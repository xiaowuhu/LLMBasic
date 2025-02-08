# 预测周期性正弦函数
# 先用 MLP 测试其不可能性
# 再用 RNN 验证其可能性
# 给出一个周期的样本数据 0~2pi

from matplotlib import pyplot as plt
import numpy as np
from common.DataLoader_14 import DataLoader_14
from common.HyperParameters import HyperParameters
import common.LearningRateScheduler as LRScheduler
from H1_1_Train_Base import train_model, load_npz_data
from H1_5_Sin_RNN_Train import RNN_Sin

def expand_test_model(data_loader:DataLoader_14, model:RNN_Sin):
    test_x, test_y = data_loader.get_test()
    # 创建时间步
    t = np.linspace(0, 6 * np.pi, 61)
    # 选择第一个测试数据
    result = []
    id = 0
    x = test_x[id:id+1]  # 原始测试数据
    plt.scatter(t[0:steps], x)
    # 把前一个测试数据的输出结果作为后一个测试的输入x
    # 而不是直接使用测试数据中的x
    for i in range(len(t) - steps):
        z = model.forward(x)
        result.append(z)
        x = np.roll(x, -1, axis=1)
        x[:,-1] = z  # 替换
    # 递归测试结果
    plt.scatter(t[steps:], result, marker='x', label="预测结果")
    plt.legend()
    plt.grid()
    plt.show()

if __name__=='__main__':
    num_input, num_hidden, num_output = 1, 2, 1
    steps = 7
    model = RNN_Sin(num_input, num_hidden, num_output, steps, optimizer="Adam")
    data_loader = load_npz_data("sin_train_steps.npz", "sin_test_steps.npz")
    params = HyperParameters(max_epoch=100, batch_size=8)
    lrs = LRScheduler.step_lrs(0.01, 0.95, 100)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_sin_rnn_steps_best")
    #training_history.show_loss(yscale="linear")
    #model.load("model_sin_rnn_11_best")
    expand_test_model(data_loader, model)