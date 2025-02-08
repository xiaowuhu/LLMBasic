import torch
import random
import torch.nn.functional as F

from H7_4_1_Data_Model import DataGenerator, SineModel


def train_maml(model: SineModel, epochs, tasks, lr_inner=0.01, lr_outer=0.001):
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=lr_inner)
    for epoch in range(epochs):
        init_param = model.get_paramaters_copy()  # 保存原始的参数
        random.shuffle(tasks)
        total_loss = 0
        for task in tasks:  # 相当于不同的 task，每个dataset可以产生不同相位和振幅的正弦波
            model.set_paramaters(init_param)
            # support set
            x_support, y_support = task.get_support_set()
            y_hat = model.forward(x_support)  # run forward pass to initialize weights
            loss = F.mse_loss(y_support, y_hat)
            loss.backward()
            optimizer_sgd.step()
            optimizer_sgd.zero_grad()
            # query set
            x_query, y_query = task.get_query_set()
            y_pred = model.forward(x_query)
            loss = F.mse_loss(y_query, y_pred)
            total_loss += loss.item()
            grad = torch.autograd.grad(loss, model.parameters())
            model.keep_grad(grad)
        
        model.set_paramaters(init_param)
        model.update_parameters(lr_outer, NUM_TASK)
        model.zero_grad()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(tasks)))

if __name__=="__main__":
    NUM_TASK = 100  # 任务数量
    k = 10
    train_datasets = [DataGenerator(k) for _ in range(NUM_TASK)]

    model = SineModel()
    epochs = 1000
    train_maml(model, epochs, train_datasets)
    torch.save(model.state_dict(), '../model/ch7/maml/sine_maml.pth')
