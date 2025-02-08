import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from H7_4_1_Data_Model import DataGenerator, SineModel

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 学习率为0.01，元学习模型优化器Adam（同maml论文）
def test_maml(model: SineModel, steps, task, lr, num_sample):
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=lr)
    results = []
    x_support, y_support = task.get_support_set()
    x_query, y_query = task.get_query_set()
    for step in range(steps):
        model.train()
        y_hat = model.forward(x_support[0:num_sample])  # run forward pass to initialize weights
        loss = F.mse_loss(y_support[0:num_sample], y_hat)
        loss.backward()
        optimizer_sgd.step()
        optimizer_sgd.zero_grad()
        model.eval()
        y_pred = model.forward(x_query)
        loss = F.mse_loss(y_query, y_pred)
        results.append(y_pred.detach().numpy())
        print(f"微调{step}步，loss {loss.item()}")
    return results


def draw_result(task, results, steps, num_sample=0):
    x_train, y_train = task.get_train_set()
    x_train = x_train[0:num_sample]
    y_train = y_train[0:num_sample]
    x_test, y_test = task.get_test_set()
    plt.scatter(x_train.numpy(), y_train.numpy(), marker='^', label='训练数据')
    plt.plot(x_test.numpy(), y_test.numpy(), label='目标曲线')
    fits = [0, 1, 9]
    ls = ['--', '-.', ':']
    j = 0
    for i in range(steps):
        if i in fits:
            plt.plot(x_test.numpy(), results[i], linestyle=ls[j], label=f'微调{i}步')
            j += 1
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

        
if __name__=="__main__":
    print("5个学习样本")
    k = 10
    task = DataGenerator(k)
    model = SineModel()
    model.load_state_dict(torch.load("../model/ch7/maml/sine_maml.pth"))
    steps = 10
    learning_rate = 0.01
    results = test_maml(model, steps, task, learning_rate, num_sample=5)
    draw_result(task, results, steps, num_sample=5)

    print("10个学习样本")
    model.load_state_dict(torch.load("../model/ch7/maml/sine_maml.pth"))
    results = test_maml(model, steps, task, learning_rate, num_sample=10)
    draw_result(task, results, steps, num_sample=10)
