import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from H7_4_1_Data_Model import SineModel, DataGenerator

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def train(model, epochs, datasets, lr, model_path):
    if os.path.exists(model_path):
        print("Model exists, load model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        return
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for dataset in datasets:
            x, y = dataset.get_train_set()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(datasets)))

    torch.save(model.state_dict(), model_path)


# 用一组数据进行迁移学习
def transfer_learning(model, steps, dataset: DataGenerator, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x_test, y_test = dataset.get_test_set()
    results = []
    for epoch in range(steps):
        y_hat = model(x_test)
        loss = F.mse_loss(y_test, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results.append(y_hat.detach().numpy())
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
    return results


def draw_result(dataset: DataGenerator, results, checkpoints):
    x_test, y_test = dataset.get_test_set()
    x_train, y_train = dataset.get_train_set()
    plt.plot(x_test.numpy(), y_test.numpy(), label="目标曲线")
    plt.scatter(x_train.numpy(), y_train.numpy(), marker='^', label="训练数据")
    ls = ['--', '-.', ':']
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
    model = SineModel()
    print("基准学习")
    epoch = 100
    k = 10
    num_dataset = 100
    learning_rate = 0.001
    datasets = [DataGenerator(k) for _ in range(num_dataset)]
    model_path = "../model/ch7/maml/sine_baseline.pth"
    train(model, epoch, datasets, learning_rate, model_path)
    print("迁移学习")
    steps = 100
    results = transfer_learning(model, steps, datasets[0], learning_rate)
    draw_result(datasets[0], results, [1, 9])
