import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
import os

from H1_6_1_Name_DataGenerator import DataLoader_train

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)
        self.hidden_0 =  torch.zeros(1, self.hidden_size)

    def forward(self, input):
        hidden = self.hidden_0
        for i in range(input.shape[1]):  # steps
            hidden = F.tanh(self.U(input[:,i]) + self.W(hidden))
            output = self.V(hidden)
        return output


def print_training_progress(epoch, num_epochs, step, total_step, step_loss, lr_scheduler):
    # todo: add validation set here
    rate = (step + 1) / total_step
    prefix = "*" * int(rate * 50)
    suffix = "." * int((1-rate) * 50)
    print("\rEpoch:{}/{} (lr={:.5f}) {:^4.0f}%[{}->{}]{:.4f}".format(
        epoch + 1, num_epochs, lr_scheduler.get_last_lr()[0],
        int(rate * 100), prefix, suffix, step_loss),
        end="")

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

def test_model(test_loader: DataLoader_train, model, device, loss_func):
    model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        test_x, test_y = test_loader.get_test_data()
        x, y = test_x.to(device), test_y.to(device)
        predict = model(x)
        loss = loss_func(predict, y) # 添加损失值
        pred = predict.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
        correct = pred.eq(y.data.view_as(pred)).cpu().sum().item()
 
    return loss.item(), correct/test_x.shape[0]

# 由于 loss 部分特殊，所以单独实现
def train_model(num_epochs, model, device, train_loader, optimizer, lr_scheduler, loss_func, name, best_correct=0.5):
    total_step = 1000
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        train_loader.shuffle()
        epoch_done = False
        step = 0
        while epoch_done == False:
            step += 1
            train_x, train_y = train_loader.get_batch()
            epoch_done = train_loader.is_epoch_done()
            x = train_x.to(device)
            y = train_y.to(device)
            predict = model(x)
            # 计算损失函数值
            step_loss = loss_func(predict, y)
            running_loss += step_loss
            # 优化器内部参数梯度必须变为0
            optimizer.zero_grad()
            # 损失值后向传播
            step_loss.backward()
            # 更新模型参数
            optimizer.step()
            print_training_progress(epoch, num_epochs, step, total_step, step_loss, lr_scheduler)
        lr_scheduler.step()
        print()
        total_step = step

        test_loss, correct_rate = test_model(train_loader, model, device, loss_func)
        print("Running Loss:{:.4f}, Val loss:{:.4f}, Accu:{:.2f}%".format(
            running_loss/step, test_loss,  correct_rate * 100))
        if correct_rate > best_correct:
            save_model(model, name)
            best_correct = correct_rate

def main(model:nn.Module, save_name:str, pretrained_model_name:str = None, best_correct:float = 0.5):
    batch_size = 8
    epoch = 100
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")  # 没有 GPU 时
    train_loader = DataLoader_train(n_categories, batch_size)
    # 初始化模型 将网络操作移动到GPU或者CPU
    model = model.to(DEVICE)
    model.hidden_0 = model.hidden_0.to(DEVICE)
    print(model)

    #加载预训练模型
    if pretrained_model_name is not None:
        load_model(model, pretrained_model_name, DEVICE)

    # 定义交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()
    # 第一步， 先使用Adam优化器训练100轮并保存最佳结果，设置best_correct为0.5
    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay=1e-6)
    # 第二步，在上个100轮的基础上，加载模型，使用SGD优化器再训练100轮
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.0001)
    # 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
    train_model(epoch, model, DEVICE, train_loader, optimizer, exp_lr_scheduler, loss_func, save_name, best_correct)


if __name__=="__main__":
    n_hidden = 64
    n_letters = 55
    n_categories = 18
    rnn = RNN(n_letters, n_hidden, n_categories)
    # 第一步
    main(rnn, "model_name_best_adam.pth", best_correct=0.5)
    # 第二步
    main(rnn, "model_name_best_sgd.pth", "model_name_best_adam.pth", best_correct=0.82)



