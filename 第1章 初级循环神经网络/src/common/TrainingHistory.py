import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 记录训练过程
class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.train_loss = []
        self.train_accu = []
        self.val_loss = []
        self.val_accu = []
        self.best_val_loss = np.inf
        self.best_val_accu = np.inf
        self.best_iteration = 0

    def append(self, iteration, train_loss, train_accu, val_loss, val_accu):
        self.iteration.append(iteration)
        self.train_loss.append(train_loss)
        self.train_accu.append(train_accu)
        self.val_loss.append(val_loss)
        self.val_accu.append(val_accu)
        self.history = None
        # 得到最小误差值对应的权重值
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_iteration = iteration

    def get_history(self):
        if self.history is None:
            history = np.vstack(
                (self.iteration, 
                 self.train_loss, 
                 self.train_accu, 
                 self.val_loss, 
                 self.val_accu))
            self.history = history.transpose()
        return self.history

    def get_best(self):
        return self.best_iteration, self.best_val_loss, self.W, self.B

    # 获得当前点的前 {10} 个误差记录
    def get_avg_loss(self, iter:int, count:int=10):
        assert(iter >=0)
        start = max(0, iter-count)
        end = iter
        return self.val_loss[start:end]

    def save_history(self, name):
        self.get_history()
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        np.savetxt(file_path, self.history, fmt="%.6f")

    def load_history(self, name):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        if os.path.exists(file_path):
            self.history = np.loadtxt(file_path)
            return self.history
        else:
            print("training history file not exist!!!")

    # 平滑处理
    def moving_average(self, data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        smooth_data = np.convolve(data, window, 'full')
        start = min(5, window_size//2)
        end = start + data.shape[0]
        return smooth_data[start:end] # 会多出 window_size-1 个元素

    def show_loss(self, start=0, smooth=None, yscale="log"):
        self.get_history()
        iteration, train_loss, train_accu, val_loss, val_accu = \
            self.history[start:,0], self.history[start:,1], self.history[start:,2], self.history[start:,3], self.history[start:,4]

        fig = plt.figure(figsize=(9, 4.5))
        # loss
        ax = fig.add_subplot(1, 2, 1)
        if smooth is not None:
            train_loss = self.moving_average(train_loss, smooth)
        ax.plot(iteration, train_loss, label='训练集')
        if smooth is not None:
            val_loss = self.moving_average(val_loss, smooth)
        ax.plot(iteration, val_loss, label='验证集', marker='o', markevery=0.3)
        ax.set_xlabel("迭代次数")
        ax.set_title("误差")
        ax.set_yscale(yscale)
        ax.legend()
        ax.grid()
        # accu
        ax = fig.add_subplot(1, 2, 2)
        if smooth is not None:
            train_accu = self.moving_average(train_accu, smooth)
        ax.plot(iteration, train_accu, label='训练集')
        if smooth is not None:
            val_accu = self.moving_average(val_accu, smooth)
        ax.plot(iteration, val_accu, label='验证集', marker='o', markevery=0.3)
        ax.set_xlabel("迭代次数")
        ax.set_title("准确率")
        ax.set_yscale(yscale)
        ax.grid()
        ax.legend()
        plt.show()        
    

    def show_loss_only(self, start=0, smooth=None):
        self.get_history()
        iteration, train_loss, train_accu, val_loss, val_accu = \
            self.history[start:,0], self.history[start:,1], self.history[start:,2], self.history[start:,3], self.history[start:,4]

        fig = plt.figure(figsize=(9, 4.5))
        # loss
        ax = fig.add_subplot(1, 1, 1)
        if smooth is not None:
            train_loss = self.moving_average(train_loss, smooth)
        ax.plot(iteration, train_loss, label='训练集')
        if smooth is not None:
            val_loss = self.moving_average(val_loss, smooth)
        ax.plot(iteration, val_loss, label='验证集', marker='o', markevery=0.3)
        ax.set_xlabel("迭代次数")
        ax.set_title("误差")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()
        plt.show()        
