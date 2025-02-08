import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt 
 
#Network设计
class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,  #[b,seq,f]
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std=0.001)
        self.linear = nn.Linear(hidden_size,output_size)
        
    def forward(self,x,hidden_prev):
        out,hidden_prev = self.rnn(x,hidden_prev)
        #[1,seq,h]=>[seq,h]
        out = out.view(-1,hidden_size)
        out = self.linear(out)  #[seq,h]=>[seq,1]
        out = out.unsqueeze(dim=0) #=>[1,seq,1]
        return out, hidden_prev

num_time_steps=50 #共50个点
input_size=1
hidden_size=16
output_size=1
lr=0.01

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr)

#Train
hidden_prev = torch.zeros(1,1,hidden_size)
for iter in range(6000):
    start = np.random.randint(10,size=1)[0]
    time_steps = np.linspace(start,start+10,num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps,1)
    x = torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
    y=torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)
    
    output,hidden_prev = model(x,hidden_prev)
    hidden_prev = hidden_prev.detach()
    
    loss = criterion(output,y)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter %100 == 0:
        print("Iteration:{} loss {}".format(iter,loss.item()))
        #print("Iteration :{ }  loss { }".format(iter,loss.item()))
 
#给test时生成一个数据
start = np.random.randint(3,size=1)[0]
time_steps = np.linspace(start,start+10,num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps,1)
x = torch.tensor(data[:-1]).float().view(1,num_time_steps -1,1)  #给0~48的数据
y = torch.tensor(data[1:]).float().view(1,num_time_steps -1,1)  #预测1~49号点的数据

#Predict
predictions = []
input = x[:,0,:]
for _ in range(x.shape[1]):
#for _ in range(x.shape[1])
    input = input.view(1,1,1)
    (pred,hidden_prev) = model(input,hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])
 
x=x.data.numpy().ravel()
y=y.data.numpy()
plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1],x.ravel())
 
plt.scatter(time_steps[1:],predictions)
plt.show()