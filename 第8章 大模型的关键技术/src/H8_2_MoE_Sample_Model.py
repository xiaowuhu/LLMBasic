# 向原作者 Shahriar Hossain 致敬
import torch
import torch.nn as nn


# 专家模型
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.softmax(self.layer2(x), dim=1)


# 门控网络
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gating, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.softmax(x, dim=1)


# MoE 模型
class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # 假设所有专家都有相同的输入维度,如果不是的话需要调整
        input_dim = trained_experts[0].layer1.in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # 从门控网络得到权重
        weights = self.gating(x)
        # 计算专家模型输出
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # 把权重输出的形状变成与专家模型的输出形状一致，便于后面做乘法
        weights = weights.unsqueeze(1).expand_as(outputs)
        # 权重与输出相乘，并把结果相加
        return torch.sum(outputs*weights, dim=2)
