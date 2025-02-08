import numpy as np
import math


# 二分类的准确率计算
def tpn2(predict, label):
    result = (np.round(predict) == label) # a <= 0.5 -> 0, a > 0.5 -> 1
    num_correct = result.sum()  # result 中是 True(分类正确),False（分类错误）, sum() 是True的个数
    return num_correct / predict.shape[0] # 分类正确的比例（正负类都算）

# 多分类的准确率计算
def tpn3(predict, label):
    ra = np.argmax(predict, axis=1)
    ry = np.argmax(label, axis=1)
    r = (ra == ry)
    correct_rate = np.mean(r)
    return correct_rate

# R2 准确率函数
def r2(y, mse_loss):
    var = np.var(y)
    accuracy = 1 - mse_loss / var
    return accuracy

# RMSE 准确率函数，越准确，该值越低
# 1 - RMSE，可以令最大值为 1
def rmse(z, y):
    rmse = math.sqrt(np.mean(np.square(z - y)))
    return 1 - rmse
