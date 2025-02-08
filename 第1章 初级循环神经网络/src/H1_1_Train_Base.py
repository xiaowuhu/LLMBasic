import os
import math
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layers
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import matplotlib.pyplot as plt
import common.LearningRateScheduler as LRScheduler


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_txt_data()
    data_loader.shuffle_data()
    #data_loader.split_data(0.8)
    return data_loader

def load_npz_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data()
    data_loader.shuffle_data()
    return data_loader


# 计算损失函数和准确率
def check_loss(
        data_loader:DataLoader_14, 
        batch_size: int, batch_id: int, 
        model: Sequential, 
        training_history:TrainingHistory, 
        epoch:int, iteration:int, 
        learning_rate:float
):
    # 训练集
    x, label = data_loader.get_batch(batch_size, batch_id)
    train_loss, train_accu = model.compute_loss_accuracy(x, label)
    # 验证集
    # x, label = data_loader.get_val()
    # val_loss, val_accu = model.compute_loss_accuracy(x, label)
    # 没有验证集，用训练集代替
    val_loss, val_accu = train_loss, train_accu
    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    print("轮数 %d, 迭代 %d, 训练集: loss %.6f, accu %.4f, 验证集: loss %.6f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    return val_loss

def train_model(
        data_loader: DataLoader_14, 
        model,
        params: HyperParameters,
        lrs: LRScheduler,
        checkpoint = 1,
        name:str = None,
):
    training_history = TrainingHistory()
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    check_iteration = int(batch_per_epoch * checkpoint)    
    iteration = 0 # 每一批样本算作一次迭代
    best_loss = 1000
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_Y = data_loader.get_batch(params.batch_size, batch_id)
            batch_Z = model.forward(batch_X)
            model.backward(batch_Z, batch_Y)
            model.update(params.learning_rate)
            iteration += 1
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                val_loss = check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
                if name is not None and val_loss < best_loss:
                    model.save(name)
                    print("save model loss=", val_loss)
                    best_loss = val_loss
    return training_history

def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, label = data_loader.get_test()
    test_loss, test_accu = model.compute_loss_accuracy(test_x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))

