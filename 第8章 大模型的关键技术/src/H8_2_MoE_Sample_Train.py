
# 向原作者 Shahriar Hossain 致敬
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from H8_2_MoE_Sample_Data import generate_data
from H8_2_MoE_Sample_Model import Expert, MoE

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


def train_model(model, x, y, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

# Evaluate all models
def evaluate(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
    return accuracy

def main():
    # 生成数据
    num_samples, input_dim, hidden_dim, output_dim = 6000, 4, 64, 3
    train_expert, train_moe, test_moe = generate_data(num_samples, input_dim)
    x_test, y_test = test_moe[0], test_moe[1]
    # 定义三个专家，并测试其初始化时的准确率
    experts = []
    expert1 = Expert(input_dim, hidden_dim, output_dim)
    experts.append(expert1)
    expert2 = Expert(input_dim, hidden_dim, output_dim)
    experts.append(expert2)
    expert3 = Expert(input_dim, hidden_dim, output_dim)
    experts.append(expert3)
    # 先测试一次训练前的准确率
    for i in range(3):
        accuracy = evaluate(experts[i], x_test, y_test)
        print(f"训练前专家 {i+1} 准确率:", accuracy)    
    
    # 训练专家模型
    epochs, learning_rate = 1000, 0.001
    for i in range(3):
        x_train, y_train = train_expert[i][0], train_expert[i][1]
        train_model(experts[i], x_train, y_train, epochs, learning_rate)
        save_model(experts[i], "expert_" + str(i+1) + ".pth")
        accuracy = evaluate(experts[i], x_test, y_test)
        print(f"训练后专家 {i+1} 准确率:", accuracy)

    # 训练 MoE 模型
    epochs, learning_rate = 500, 0.001
    x_train, y_train = train_moe[0], train_moe[1]
    moe_model = MoE([expert1, expert2, expert3])
    train_model(moe_model, x_train, y_train, epochs, learning_rate)
    save_model(moe_model, "moe.pth")

    # 测试 MoE 模型
    accuracy_moe = evaluate(moe_model, x_test, y_test)
    print("MoE 准确率:", accuracy_moe)

def test():
    device = "cpu"
    input_dim, hidden_dim, output_dim = 4, 64, 3
    expert1 = Expert(input_dim, hidden_dim, output_dim)
    load_model(expert1, "expert_1.pth", device)
    expert2 = Expert(input_dim, hidden_dim, output_dim)
    load_model(expert2, "expert_2.pth", device)
    expert3 = Expert(input_dim, hidden_dim, output_dim)
    load_model(expert3, "expert_3.pth", device)
    moe_model = MoE([expert1, expert2, expert3])
    load_model(moe_model, "moe.pth", device)

    train_expert, train_moe, test_moe = generate_data(6000, input_dim)
    x_test, y_test = test_moe[0], test_moe[1]
    accuracy_moe = evaluate(moe_model, x_test[0:1], y_test[0:1])
    print(accuracy_moe)

if __name__=="__main__":
    main()
    test()
