import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# 定义教师模型和学生模型
teacher_model = models.resnet18(pretrained=True)
student_model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 10)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

# 训练数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 蒸馏过程
for epoch in range(10):
    running_loss_teacher = 0.0
    running_loss_student = 0.0
    
    for inputs, labels in trainloader:
        # 教师模型的前向传播
        outputs_teacher = teacher_model(inputs)
        #loss_teacher = criterion(outputs_teacher, labels)
        #running_loss_teacher += loss_teacher.item()
        
        # 学生模型的前向传播
        outputs_student = student_model(inputs)
        loss_student = criterion(outputs_student, labels) + 0.1 * torch.sum((outputs_teacher - outputs_student) ** 2)
        running_loss_student += loss_student.item()
        
        # 反向传播和参数更新
        #optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()
        #loss_teacher.backward()
        #optimizer_teacher.step()
        loss_student.backward()
        optimizer_student.step()
    
    #print(f'Epoch {epoch+1}/10 \t Loss Teacher: {running_loss_teacher / len(trainloader)} \t Loss Student: {running_loss_student / len(trainloader)}')
    print(f'Epoch {epoch+1}/10 \t Loss Student: {running_loss_student / len(trainloader)}')
