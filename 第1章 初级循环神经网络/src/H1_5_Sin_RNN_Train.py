# 预测周期性正弦函数
# 先用 MLP 测试其不可能性
# 再用 RNN 验证其可能性
# 给出一个周期的样本数据 0~2pi

import numpy as np
import common.Layers as layers
from common.HyperParameters import HyperParameters
import common.LearningRateScheduler as LRScheduler
from common.Estimators import rmse
from H1_1_Train_Base import train_model, load_npz_data


class RNN_Sin(layers.Operator):
    def __init__(self, num_input, num_hidden, num_output,
                 steps,
                 init_method: str='normal',
                 optimizer: str="SGD",
    ):
        self.steps = steps
        self.U = layers.WeightsBias(num_input, num_hidden, init_method, optimizer)
        self.W = layers.WeightsBias(num_hidden, num_hidden, init_method, optimizer)
        self.V = layers.WeightsBias(num_hidden, num_output, init_method, optimizer)
        # Tanh的正向计算时记录了结果a在反向时使用，所以分开成4个，而不能共用一个
        self.tanh = []
        self.x = [None] * steps
        self.h = [None] * steps
        self.s = [None] * steps
        self.ds = [None] * steps
        self.dh = [None] * steps
        self.dU = [None] * steps
        self.dW = [None] * steps
        self.dW[0] = np.zeros_like(self.W.Weight)
        for t in range(steps):
            self.tanh.append(layers.Tanh())
        self.loss_func = layers.MSE()
        self.set_parameters(self.U, self.V, self.W)

    def forward(self, X):
        self.batch_size = X.shape[0]
        for t in range(self.steps):
            self.x[t] = X[:, t]
            # 式（1.4.1）
            self.h[t] = np.dot(self.x[t], self.U.Weight)+ self.U.Bias
            if t > 0:
                self.h[t] += np.dot(self.s[t-1], self.W.Weight) + self.W.Bias
            self.s[t] = self.tanh[t](self.h[t])
        # 只在最后一个时间步有输出
        self.z = np.dot(self.s[t], self.V.Weight) + self.V.Bias
        return self.z

    def backward(self, Z, Y):
        # 只有最后一个时间步有反向误差 dz
        dz = Z - Y  # 式（1.4.2）
        for t in range(self.steps-1, -1, -1):
            if t == self.steps - 1:  # 式（1.4.3）
                ds = np.dot(dz, self.V.Weight.T)
                # 式（1.4.6）
                self.V.dW = np.dot(self.s[t].T, dz)
                self.V.dB = np.mean(dz, keepdims=True)
            else:  # 式（1.4.4）
                ds = np.dot(self.dh[t+1], self.W.Weight.T)
            # 式（1.4.5）
            self.dh[t] = self.tanh[t].backward(ds)
            # 式（1.4.7）
            self.dU[t] = np.dot(self.x[t].T, self.dh[t])
            # 式（1.4.8）
            if t > 0:
                self.dW[t] = np.dot(self.s[t-1].T, self.dh[t])
        self.U.dW = self._weight_grad(self.dU)  # 求均值
        self.U.dB = self._bias_grad(self.dh)    # 求均值
        self.W.dW = self._weight_grad(self.dW)  # 求均值
        self.W.dB = self._bias_grad(self.dh)    # 求均值

    def _weight_grad(self, list_grad):
        array_grad = np.array(list_grad)
        grad = np.sum(array_grad, axis=0) / self.batch_size / self.steps
        return grad

    def _bias_grad(self, list_grad):
        array_grad = np.array(list_grad)
        grad = np.sum(array_grad, axis=(0,1), keepdims=True)
        grad = np.squeeze(grad, axis=0) / self.batch_size / self.steps
        return grad

    def compute_loss_accuracy(self, x, y):
        z = self.forward(x)
        loss = self.loss_func(z, y)
        accu = rmse(z, y)
        return loss, accu

    def set_parameters(self, *params):
        self.paramters_dict = {}
        unique_id = 0
        for wb in params:
            op_name = self.__class__.__name__ + "_" + str(unique_id)
            if wb is not None:
                self.paramters_dict[op_name] = wb
            unique_id += 1
    
    def update(self, lr):
        for _, WB in self.paramters_dict.items():
            if isinstance(WB, tuple):
                for wb in WB:
                    wb.Update(lr)
            else: # WeightsBias object
                WB.Update(lr)        

    def load(self, name):
        U = super().load_from_txt_file(name + "_U")
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        self.U.set_WB_value(U)

        W = super().load_from_txt_file(name + "_W")
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        self.W.set_WB_value(W)

        V = super().load_from_txt_file(name + "_V")
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        self.V.set_WB_value(V)

    def save(self, name):
        super().save_to_txt_file(name + "_U", self.U.get_WB_value())
        super().save_to_txt_file(name + "_W", self.W.get_WB_value())
        super().save_to_txt_file(name + "_V", self.V.get_WB_value())


if __name__=='__main__':
    num_input, num_hidden, num_output = 1, 2, 1
    steps = 20
    model = RNN_Sin(num_input, num_hidden, num_output, steps, optimizer="Adam")
    data_loader = load_npz_data("sin_train_20.npz", "sin_test_20.npz")
    params = HyperParameters(max_epoch=100, batch_size=8)
    lrs = LRScheduler.step_lrs(0.01, 0.95, 100)
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1) #, name="model_sin_rnn_best")
    training_history.show_loss(yscale="linear")
