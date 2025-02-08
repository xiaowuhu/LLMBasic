
import numpy as np
import common.Optimizers as Optimizers

class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, optimizer):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.Weight, self.Bias = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.dW = np.zeros_like(self.Weight).astype(np.float32)
        self.dB = np.zeros_like(self.Bias).astype(np.float32)
        self.opt_W = Optimizers.Optimizer.create_optimizer(optimizer)
        self.opt_B = Optimizers.Optimizer.create_optimizer(optimizer)

    def Update(self, lr):
        self.Weight = self.opt_W.update(lr, self.Weight, self.dW)
        self.Bias = self.opt_B.update(lr, self.Bias, self.dB)

    def get_WB_value(self):
        return np.concatenate((self.Weight, self.Bias))

    def get_dWB_value(self):
        return np.concatenate((self.dW, self.dB))

    def set_WB_value(self, WB):
        self.Weight = WB[0:-1, :]
        self.Bias = WB[-1:, :]

    def set_dWB_value(self, dWB):
        self.dW = dWB[0:-1, :]
        self.dB = dWB[-1:, :]

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == "zero":
            W = np.zeros((num_input, num_output)).astype(np.float32)
        elif method == "normal":
            W = np.random.normal(size=(num_input, num_output)).astype(np.float32)
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output)).astype(np.float32)
        elif method == "xavier":
            # xavier
            W = np.random.uniform(
                -np.sqrt(6/(num_output + num_input)),
                np.sqrt(6/(num_output + num_input)),
                size=(num_input, num_output).astype(np.float32)
            )
        else:
            raise Exception("Unknown init method")
        B = np.zeros((1, num_output)).astype(np.float32)
        return W, B
