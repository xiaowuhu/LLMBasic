import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V

from H7_4_1_Models import SineModel, SineWaveTask
from H7_4_3_Transfer_Learning import copy_sine_model


def sine_fit1(net, dataset, optim):
    net.train()
    optim.zero_grad()
    x, y = dataset.training_set(force_new=False)
    y_hat = net(V(x[:, None]))
    loss = F.mse_loss(y_hat, V(y).unsqueeze(1))
    loss.backward(create_graph=False, retain_graph=True)
    optim.step()
    return loss.data.cpu().numpy()#[0]


def eval_sine_test(model, test, fits=(0, 1), lr=0.01):
    xtest, ytest = test.test_set()
    model = copy_sine_model(model)
    # Not sure if this should be Adam or SGD.
    optim = torch.optim.SGD(model.params(), lr)
        
    def get_loss(res):
        return F.mse_loss(res, V(ytest[:, None])).cpu().data.numpy()#[0]
    
    fit_res = []
    if 0 in fits:
        results = model(V(xtest[:, None]))
        fit_res.append((0, results, get_loss(results)))
    for i in range(np.max(fits)):
        sine_fit1(model, test, optim)
        if i + 1 in fits:
            results = model(V(xtest[:, None]))
            fit_res.append(
                (
                    i + 1, 
                    results,
                    get_loss(results)
                )
            )

    return fit_res


def plot_sine_test(model, test, fits, lr=0.01):
    xtest, ytest = test.test_set()
    xtrain, ytrain = test.training_set()

    fit_res = eval_sine_test(model, test, fits, lr)
    
    train, = plt.plot(xtrain.numpy(), ytrain.numpy(), '^')
    ground_truth, = plt.plot(xtest.numpy(), ytest.numpy())
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(xtest.numpy(), res.cpu().data.numpy()[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.grid()
    plt.show()

def plot_sine_learning(models, fits=(0, 1), lr=0.01):
    len_models = len(models)
    data = {'model': [], 'fits': [], 'loss': [], 'set': []}
    legend = []
    for name, model in models:
        legend.append(name)
        if not isinstance(model, list):
            model = [model]
        for n_model, model in enumerate(model):
            for n_test, test in enumerate(test_dataset):
                n_test = n_model * len(test_dataset) + n_test
                fit_res = eval_sine_test(model, test, fits, lr)
                for n, _, loss in fit_res:
                    data['model'].append(name)
                    data['fits'].append(n)
                    data['loss'].append(loss)
                    data['set'].append(n_test)

    loss = np.array(data['loss']).reshape(len_models, len(test_dataset), len(fits)) # 3,1000,10
    plots = []
    markers = ['s', 'o', '^']
    for i in range(len_models):
        cur, = plt.plot(loss[i].mean(axis=0), marker=markers[i])
        plots.append(cur)
    plt.legend(plots, legend)
    plt.grid()
    plt.show()

                    
if __name__=="__main__":
    ONE_SIDED_EXAMPLE = None
    while ONE_SIDED_EXAMPLE is None:
        cur = SineWaveTask()
        x, _ = cur.training_set()
        x = x.numpy()
        if np.max(x) < 0 or np.min(x) > 0:
            ONE_SIDED_EXAMPLE = cur

    TRAIN_SIZE = 10000
    train_dataset = [SineWaveTask() for _ in range(TRAIN_SIZE)]
    TEST_SIZE = 100
    test_dataset = [SineWaveTask() for _ in range(TEST_SIZE)]


    transfer_model: SineModel = SineModel()
    name_param = torch.load("../model/ch7/meta/sine_transfer.pth")
    for name, param in name_param:
        transfer_model.set_param(name, param)

    maml_model = SineModel()
    name_param = torch.load("../model/ch7/meta/sine_maml.pth")
    for name, param in name_param:
        maml_model.set_param(name, param)

    plot_sine_test(maml_model, test_dataset[0], fits=[0, 1, 10], lr=0.01)

    plot_sine_learning(
        [('Transfer', transfer_model), ('MAML', maml_model), ('Random', SineModel())],
        list(range(10)),
    )

    plot_sine_test(maml_model, ONE_SIDED_EXAMPLE, fits=[0, 1, 10], lr=0.01)
