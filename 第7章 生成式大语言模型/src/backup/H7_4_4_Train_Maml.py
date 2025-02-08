import torch
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.autograd import Variable as V
import copy

from H7_4_1_Models import SineModel, SineWaveTask


def get_grad(net, dataset):
    net.train()
    x, y = dataset.training_set(force_new=True)
    y_hat = net(V(x[:, None]))
    loss = F.mse_loss(y_hat, V(y).unsqueeze(1))
    loss.backward(create_graph=True, retain_graph=True)
    return
    # return loss.data.cpu().numpy()#[0]


def maml_sine(model, epochs, lr_inner=0.01, batch_size=1):
    optimizer = torch.optim.Adam(model.params())
    
    for _ in tqdm(range(epochs)):
        # Note: the paper doesn't specify the meta-batch size for this task,
        # so I just use 1 for now.
        for i, dataset in enumerate(random.sample(train_dataset, len(train_dataset))):
            new_model = SineModel()
            new_model.copy(model, same_var=True)
            get_grad(new_model, dataset)
            for name, param in new_model.named_params():
                grad = param.grad
                new_model.set_param(name, param - lr_inner * grad)
                        
            get_grad(new_model, dataset)

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
    
    torch.save(model.named_params(), "../model/ch7/meta/sine_maml.pth")
                    
if __name__=="__main__":
    TRAIN_SIZE = 10000
    train_dataset = [SineWaveTask() for _ in range(TRAIN_SIZE)]
    maml_model = SineModel()
    maml_sine(maml_model, 4) #4
