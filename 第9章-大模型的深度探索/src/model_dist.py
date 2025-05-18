import torch
import torch.nn as nn
import torch.optim as optim

# 定义教师模型和学生模型
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# 初始化模型
teacher_model = TeacherModel()
student_model = StudentModel()

# 加载数据集
# 假设 data_loader 是一个数据加载器
data_loader = ...

# 定义损失函数和优化器
criterion = nn.KLDivLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 蒸馏训练
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # 前向传播教师模型
        teacher_outputs = teacher_model(inputs)
        teacher_probs = torch.softmax(teacher_outputs, dim=1)

        # 前向传播学生模型
        student_outputs = student_model(inputs)
        student_log_probs = torch.log_softmax(student_outputs, dim=1)

        # 计算软损失
        soft_loss = criterion(student_log_probs, teacher_probs)

        # 计算硬损失
        hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)

        # 总损失
        loss = 0.5 * soft_loss + 0.5 * hard_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")