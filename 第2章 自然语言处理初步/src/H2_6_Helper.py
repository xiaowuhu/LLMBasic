
import torch
from torch import nn
import os

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))/size(y)

def evaluate_accuracy_gpu(net, data_iter, device):
    net.eval()  # 设置为评估模式
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, Y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            Y = Y.to(device)
            metric.add(accuracy(net(X), Y), size(Y))
    return metric[0] / metric[1]

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))

def test_model(test_loader, model, device, loss_func):
    print("testing...")
    model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        for test_x, test_y in test_loader:
            x, y = test_x.to(device), test_y.to(device)
            predict = model(x)
            loss += loss_func(predict, y)
            pred = predict.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
 
    return loss.item()/len(test_loader), correct/len(test_loader.dataset)


def train(model, train_data, test_data, loss_func, optimizer, num_epochs, device, save_model_name=None):
    model = model.to(device)
    best_acc = 0
    running_loss = 0
    all_loss = []
    for epoch in range(num_epochs):
        model.train()
        for iter, (X, Y) in enumerate(train_data):
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            step_loss = loss_func(output, Y)
            running_loss += step_loss
            train_accu = accuracy(output, Y)
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
            if iter % 100 == 0:
                print("epoch %d, iter %d, loss %.3f, accu %.3f" % (epoch, iter, step_loss.item(), train_accu))
                all_loss.append(running_loss.to("cpu").detach().numpy() / 100)
                running_loss = 0
        test_loss, test_acc = test_model(test_data, model, device, loss_func)            
        if save_model_name is not None:
            if test_acc > best_acc:
                best_acc = test_acc
                save_model(model, save_model_name)
        print("---- epoch %d, test loss %.3f, test acc %.3f" % (epoch, test_loss, test_acc))
    return all_loss

def predict_sentiment(net, vocab, sequence, DEVICE):
    sequence = torch.tensor(vocab[sequence.split()]).to(DEVICE)
    output = net(sequence.reshape(1, -1))
    label = torch.argmax(output, dim=1)
    print('positive' if label == 1 else 'negative')
    return 'positive' if label == 1 else 'negative'
