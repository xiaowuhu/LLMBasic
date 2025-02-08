
import numpy as np

class Optimizer(object):
    # lr 可能因为会变，所以要从外面传进来
    def update(self, lr, theta, grad):
        pass

    @staticmethod
    def create_optimizer(optimizer):
        param = None
        if isinstance(optimizer, tuple):
            assert(len(optimizer) == 2)
            name = optimizer[0]
            param = optimizer[1]
        elif isinstance(optimizer, str):
            name = optimizer
        else:
            raise
            
        if name == "SGD":
            return SGD()
        elif name == "Momentum":
            return Momentum() if param is None else Momentum(param)
        elif name == "AdaGrad":
            return AdaGrad() if param is None else AdaGrad(param)
        elif name == "RMSProp":
            return RMSProp() if param is None else RMSProp(param)
        elif name == "Adam":
            return Adam()
        else:
            raise Exception("Unknown optimizer")

class SGD(Optimizer):
    def update(self, lr: float, theta, grad):
        theta = theta - lr * grad
        return theta

class Momentum(Optimizer):
    def __init__(self, alpha=0.8):
        self.vt = 0
        self.alpha = alpha

    def update(self, lr: float, theta, grad):
        vt_new = self.alpha * self.vt - lr * grad
        theta = theta + vt_new
        self.vt = vt_new
        return theta

class AdaGrad(Optimizer):
    def __init__(self):
        self.eps = 1e-6
        self.r = 0  # 记录历史梯度的平方和

    def update(self, lr, theta, grad):
        self.r = self.r + np.multiply(grad, grad)
        theta = theta - lr / (np.sqrt(self.r) + self.eps) * grad
        return theta
    
class AdaDelta(Optimizer):
    def __init__(self):
        self.eps = 1e-5
        self.r = 0
        self.s = 0
        self.alpha = 0.9

    def update(self, lr, theta, grad):
        self.s = self.alpha * self.s + (1-self.alpha) * np.multiply(grad, grad)
        d_theta = np.sqrt((self.eps + self.r)/(self.eps + self.s)) * grad
        theta = theta - d_theta
        self.r = self.alpha * self.r + (1-self.alpha) * np.multiply(d_theta, d_theta)
        return theta


class RMSProp(Optimizer):
    def __init__(self, alpha=0.5):
        self.alpha = alpha # 0.5, 0.9, 0.99  # 衰减速率
        self.eps = 1e-6
        self.r = 0

    def update(self, lr, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.r = self.alpha * self.r + (1-self.alpha) * grad2
        theta = theta - lr / np.sqrt(self.r + self.eps) * grad
        return theta

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = 1
        self.beta2_t = 1
        self.eps = 1e-8
        self.t = 0
        self.m = 0
        self.v = 0

    # def update(self, lr, theta, grad):
    #     self.t = self.t + 1
    #     self.m = self.beta1 * self.m + (1-self.beta1) * grad
    #     self.v = self.beta2 * self.v + (1-self.beta2) * np.multiply(grad, grad)
    #     m_hat = self.m / (1 - self.beta1 ** self.t)
    #     v_hat = self.v / (1 - self.beta2 ** self.t)
    #     d_theta = m_hat / (self.eps + np.sqrt(v_hat))
    #     theta = theta - lr * d_theta
    #     return theta

    def update(self, lr, theta, grad):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1-self.beta1) * grad
        self.v = self.beta2 * self.v + (1-self.beta2) * np.multiply(grad, grad)
        self.beta1_t = self.beta1_t * self.beta1 # 避免计算乘方
        self.beta2_t = self.beta2_t * self.beta2
        m_hat = self.m / (1 - self.beta1_t)
        v_hat = self.v / (1 - self.beta2_t)
        d_theta = m_hat / (self.eps + np.sqrt(v_hat))
        theta = theta - lr * d_theta
        return theta
