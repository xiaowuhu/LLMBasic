# 预测周期性正弦函数
# 先用 MLP 测试其不可能性
# 再用 RNN 验证其可能性
# 给出一个周期的样本数据 0~2pi

from matplotlib import pyplot as plt
import numpy as np
from common.DataLoader_14 import DataLoader_14
from common.HyperParameters import HyperParameters
import common.LearningRateScheduler as LRScheduler
from H1_1_Train_Base import load_npz_data
from H1_5_Sin_RNN_Train import RNN_Sin

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def test_model(data_loader:DataLoader_14, model:RNN_Sin):
    test_x, test_y = data_loader.get_test()
    loss, accu = model.compute_loss_accuracy(test_x, test_y)
    print("loss:", loss)
    print("accu:", accu)
    # 画出所有测试数据
    t = np.linspace(0, 4 * np.pi, 2*steps+1)
    # 选其中一个测试
    ids = [0, 7, 13, 19]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6)) 
    for i, id in enumerate(ids):
        ax = axes[i//2,i%2]
        x = test_x[id:id+1]
        z = model.forward(x)
        ax.scatter(t[id:id+20], x)
        ax.scatter(t[id+20:id+21],z,marker='x')
        ax.grid()
    plt.show()

def expand_test_model(data_loader:DataLoader_14, model:RNN_Sin):
    test_x, test_y = data_loader.get_test()
    # 创建时间步
    t = np.linspace(0, 6 * np.pi, 3*steps+1)
    # 选择第一个测试数据
    result = []
    id = 0
    x = test_x[id:id+1]
    # 画出第一个测试样本
    plt.scatter(t[id:id+20], x, label="测试数据")
    # 画出所有的test_y
    plt.scatter(t[20:40], test_y, marker='^', label="测试标签")
    # 预测第一个测试样本
    z = model.forward(x)
    result.append(z)
    # 把前一个测试数据的输出结果作为后一个测试的输入x
    # 而不是直接使用测试数据中的x
    for id in range(1, 2*steps):
        x = np.roll(x, -1, axis=1)
        x[:,-1] = z  # 替换
        z = model.forward(x)
        result.append(z)
    # 递归测试结果
    plt.scatter(t[20:20+len(result)], result, marker='x', label="预测结果")
    plt.legend()
    plt.grid()
    plt.show()


if __name__=='__main__':
    num_input, num_hidden, num_output = 1, 2, 1
    steps = 20
    model = RNN_Sin(num_input, num_hidden, num_output, steps, optimizer="Adam")
    data_loader = load_npz_data("sin_train_20.npz", "sin_test_20.npz")
    params = HyperParameters(max_epoch=100, batch_size=8)
    lrs = LRScheduler.step_lrs(0.01, 0.95, 100)
    #training_history = train_model(data_loader, model, params, lrs, checkpoint=1, name="model_sin_rnn_best")
    #training_history.show_loss(yscale="linear")
    model.load("model_sin_rnn_best")
    test_model(data_loader, model)
    expand_test_model(data_loader, model)


