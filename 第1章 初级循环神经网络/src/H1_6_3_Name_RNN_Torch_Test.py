from matplotlib import ticker
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

from H1_6_1_Name_DataGenerator import DataLoader_train, nameToTensor
from H1_6_2_Name_RNN_Torch_Train import RNN


def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))


def eval_model(train_loader, model, model_name):
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    model = model.to(DEVICE)
    model.hidden_0 = model.hidden_0.to(DEVICE)
    load_model(model, model_name, DEVICE)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    total_sample = 0
    # 混淆矩阵
    confusion_matrix = torch.zeros((n_categories, n_categories))

    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        epoch_done = False
        step = 0
        while epoch_done == False:
            step += 1
            train_x, train_y = train_loader.get_batch()
            total_sample += train_x.shape[0]
            epoch_done = train_loader.is_epoch_done()
            x, y = train_x.to(DEVICE), train_y.to(DEVICE)
            predict = model(x)
            loss += loss_func(predict, y) # 添加损失值
            pred = predict.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
            for i in range(y.shape[0]):
                confusion_matrix[y[i], pred[i]] += 1
    test_loss = loss/step
    accu = correct / total_sample
    print('Test Loss: {:.4f}, Accu: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_sample, 100. * accu))


    names = train_loader.get_category_name()
    ticks = [''] + names
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cax = ax.matshow(confusion_matrix, cmap='autumn_r')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, "%d"%(confusion_matrix[i, j]), ha='center', va='center')
    set_ax(ax, ticks)

    ax = fig.add_subplot(122)
    for i in range(n_categories):
        confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()   # 归一化
    cax = ax.matshow(confusion_matrix)
    set_ax(ax, ticks)
    fig.colorbar(cax)
    plt.show()

def set_ax(ax, ticks):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)
    ax.set_ylabel("真实值")
    ax.set_xlabel("预测值")

def predict(train_loader, model, model_name, predict_names):
    DEVICE = torch.device("cpu")
    model = model.to(DEVICE)
    model.hidden_0 = model.hidden_0.to(DEVICE)
    load_model(model, model_name, DEVICE)
    all_categories = train_loader.get_category_name()

    with torch.no_grad():
        for name in predict_names:
            name_tensor = nameToTensor(name).unsqueeze(0)
            name_tensor = name_tensor.to(DEVICE)
            output = model(name_tensor)
            topv, topi = output.topk(3, 1, True)
            predictions = []
            for i in range(3):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                predictions.append([value, all_categories[category_index]])
            print('%s: (%.2f) %s, (%.2f) %s, (%.2f) %s' % (name, 
                                     predictions[0][0], predictions[0][1],
                                     predictions[1][0], predictions[1][1],
                                     predictions[2][0], predictions[2][1]))

if __name__=="__main__":
    n_hidden = 64
    n_letters = 55
    n_categories = 18
    batch_size = 32
    rnn = RNN(n_letters, n_hidden, n_categories)
    train_loader = DataLoader_train(n_categories, batch_size)
    predict_names = ["Dovesky", "Hu", "Dupre", "Gates", "Huang", "Hung", "Hun"]
    eval_model(train_loader, rnn, "model_name_best_sgd.pth")
    predict(train_loader, rnn, "model_name_best_sgd.pth", predict_names)
