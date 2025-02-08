import matplotlib.pyplot as plt

class LearningRateScheduler(object):
    def __init__(self) -> None:
        pass

    def get_learning_rate(self, iteration: int) -> float:
        pass

class fixed_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float) -> None:
        self.base_lr = base_lr

    def get_learning_rate(self, iteration: int) -> float:
        return self.base_lr

# Step 方法
class step_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float, decay_rate: float=0.9, decay_step: int=50) -> None:
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def get_learning_rate(self, iteration: int) -> float:
        return self.base_lr * (self.decay_rate ** (iteration // self.decay_step))


class multi_step_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float, decay_rate: float=0.9, decay_steps: list=[50, 100, 150]) -> None:
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_learning_rate(self, iteration: int) -> float:
        for i, step in enumerate(self.decay_steps):
            if iteration < step:
                return self.base_lr * (self.decay_rate ** i)
        return self.base_lr * (self.decay_rate ** len(self.decay_steps))


class exponential_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float, decay_rate: float=0.9) -> None:
        self.base_lr = base_lr
        self.decay_rate = decay_rate

    def get_learning_rate(self, iteration: int) -> float:
        return self.base_lr * (self.decay_rate ** iteration)


class inv_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float, gamma: float=0.001, power: float=0.75) -> None:
        self.base_lr = base_lr
        self.gamma = gamma
        self.power = power

    def get_learning_rate(self, iteration: int) -> float:
        return self.base_lr * (1 + self.gamma * iteration) ** (-self.power)


class poly_lrs(LearningRateScheduler):
    def __init__(self, base_lr: float, max_iter: int, power: float=0.9) -> None:
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power

    def get_learning_rate(self, iteration: int) -> float:
        return self.base_lr * (1 - iteration / self.max_iter) ** self.power

if __name__ == "__main__":
    a = []

    lrs = step_lrs(0.3, 0.9, 100)
    for i in range(7500):
        a.append(lrs.get_learning_rate(i))


    # lrs = multi_step_lrs(0.1, 0.9, [5, 10, 15, 50])
    # for i in range(100):
    #     print(lrs.get_learning_rate(i))


    # lrs = exponential_lrs(0.1, 0.99)
    # for i in range(100):
    #     print(lrs.get_learning_rate(i))
    
    # lrs = inv_lrs(0.1, 0.5, 0.05)
    # a = []
    # for i in range(1000):
    #     a.append(lrs.get_learning_rate(i))

    # lrs = poly_lrs(0.1, 1000, 15)
    # for i in range(1000):
    #     a.append(lrs.get_learning_rate(i))


    plt.plot(a)
    plt.show()