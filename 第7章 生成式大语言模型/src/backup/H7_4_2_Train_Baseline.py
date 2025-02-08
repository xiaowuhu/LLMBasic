import torch
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.autograd import Variable as V

from H7_4_1_Models import SineModel, SineWaveTask

def sin_train(net, dataset, optim):
    net.train()
    optim.zero_grad()
    x, y = dataset.training_set(force_new=False)
    y_hat = net(V(x[:, None]))
    loss = F.mse_loss(y_hat, V(y).unsqueeze(1))
    loss.backward(create_graph=False, retain_graph=True)
    optim.step()
    return loss.data.cpu().numpy()#[0]

def train_baseline(epochs):
    optim = torch.optim.Adam(sin_baseline_model.params())

    for _ in tqdm(range(epochs)):
        for t in random.sample(train_dataset, len(train_dataset)):
            sin_train(sin_baseline_model, t, optim)
    
    torch.save(sin_baseline_model.named_params(), "../model/ch7/meta/sine_baseline.pth")

if __name__=="__main__":
    TRAIN_SIZE = 10000  # 10000
    train_dataset = [SineWaveTask() for _ in range(TRAIN_SIZE)]
    sin_baseline_model: SineModel = SineModel()
    train_baseline(3)
