import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V

from H7_4_1_Models import SineModel, SineWaveTask
from H7_4_2_Train_Baseline import sin_train


def copy_sine_model(model):
    m = SineModel()
    m.copy(model)
    return m

def transfer_learning(model, test, fits=(0, 1), lr=0.01):
    xtest, ytest = test.test_set()
    transfer_model = copy_sine_model(model)
    # Not sure if this should be Adam or SGD.
    optim = torch.optim.SGD(transfer_model.params(), lr)
        
    def get_loss(res):
        return F.mse_loss(res, V(ytest[:, None])).cpu().data.numpy()#[0]
    
    fit_res = []
    if 0 in fits:
        results = transfer_model(V(xtest[:, None]))
        fit_res.append((0, results, get_loss(results)))
    for i in range(np.max(fits)):
        sin_train(transfer_model, test, optim)
        if i + 1 in fits:
            results = transfer_model(V(xtest[:, None]))
            fit_res.append(
                (
                    i + 1, 
                    results,
                    get_loss(results)
                )
            )
    torch.save(transfer_model.named_params(), "../model/ch7/meta/sine_transfer.pth")
    return fit_res

def plot_sine_test(baseline_model, test_dataset, fits=(0, 1), lr=0.01):
    xtest, ytest = test_dataset.test_set()
    fit_res = transfer_learning(baseline_model, test_dataset, fits, lr)
    
    ground_truth, = plt.plot(xtest.numpy(), ytest.numpy())
    plots = [ground_truth]
    legend = ['True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(xtest.numpy(), res.cpu().data.numpy()[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.grid()
    plt.show()

if __name__=="__main__":
    TEST_SIZE = 1000
    test_dataset = [SineWaveTask() for _ in range(TEST_SIZE)]
    sin_baseline_model: SineModel = SineModel()
    named_params = torch.load("../model/ch7/meta/sine_baseline.pth")
    for name, param in named_params:
        sin_baseline_model.set_param(name, param)

    plot_sine_test(sin_baseline_model, test_dataset[0], fits=[0, 1, 10], lr=0.02)
