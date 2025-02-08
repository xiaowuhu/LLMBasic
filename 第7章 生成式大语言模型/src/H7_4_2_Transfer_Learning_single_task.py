import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from H7_4_1_Data_Model import SineModel, DataGenerator

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def train(model, epochs, dataset, lr, model_path):
    if os.path.exists(model_path):
        print("Model exists, load model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        return
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        x, y = dataset.get_train_set()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss))

    torch.save(model.state_dict(), model_path)

def draw_result1(model, dataset1: DataGenerator):
    x_test, y_test = dataset1.get_test_set()
    plt.plot(x_test.numpy(), y_test.numpy(), label="基准曲线")
    y_pred = model(x_test)
    loss = F.mse_loss(y_pred, y_test)
    print("Loss = ", loss.item())
    plt.plot(x_test.numpy(), y_pred.detach().numpy(), linestyle=':', label="预测数据")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


# 用一组数据进行迁移学习
def transfer_learning(model, steps, dataset2: DataGenerator, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x_test, y_test = dataset2.get_test_set()
    results = []
    for step in range(steps):
        y_hat = model(x_test)
        loss = F.mse_loss(y_test, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results.append(y_hat.detach().numpy())
        print("Step: {}, Loss: {}".format(step, loss.item()))
    return results


def draw_result2(dataset1: DataGenerator, dataset2: DataGenerator, results, checkpoints):
    x_test, y_test = dataset1.get_test_set()
    plt.plot(x_test.numpy(), y_test.numpy(), label="基准曲线")

    x_test, y_test = dataset2.get_test_set()
    x_train, y_train = dataset2.get_train_set()
    plt.plot(x_test.numpy(), y_test.numpy(), label="目标曲线", linestyle=":")
    ls = ['--', '-.']
    j = 0
    for i in range(len(results)):
        if i in checkpoints:
            plt.plot(x_test.numpy(), results[i], linestyle=ls[j], label="学习{}步".format(i))
            j += 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


if __name__=="__main__":
    # 训练一个标准的正弦曲线回归模型
    model = SineModel()
    epoch = 100
    k = 1000
    dataset1 = DataGenerator(k, amp=1.0, phase=0.0)
    learning_rate = 0.01
    model_path = "../model/ch7/maml/sine_single_task.pth"
    train(model, epoch, dataset1, learning_rate, model_path)
    draw_result1(model, dataset1)

    # 用一个任务进行迁移学习
    k = 100
    dataset2 = DataGenerator(k, amp=2.0, phase=0.5)
    steps = 10
    learning_rate = 0.01
    results = transfer_learning(model, steps, dataset2, learning_rate)
    draw_result2(dataset1, dataset2, results, [1, 9])
