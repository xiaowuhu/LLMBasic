from multiprocessing import shared_memory
import numpy as np

from .Layers import Classifier, Operator as op, LossFunctions as loss
from .Estimators import r2, tpn2, tpn3

# 模型基类
class Module(object):
    # def predict(self, X):
    #     return self.forward(X)

    # 目前这个operators必须有,否则从locals()中获取不到
    def save_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id)
            op.save(op_name)
            unique_id += 1

    def load_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id)
            op.load(op_name)
            unique_id += 1


# 顺序计算类
class Sequential(Module):
    def __init__(self, *operators):
        self.operator_seq:list[op.Operator] = list(operators)
        self.reversed_ops = self.operator_seq[::-1]
        self.loss_function = None
        self.classifier_loss_function = None
        self.classifier_function = None
        self.net_type = "Regression"  # Classifier | BinaryClassifier
        self.paramters_dict = self.get_parameters()

    # 添加一个操作符 layer，顺序由调用者指定
    def add_op(self, operator):
        if len(self.operator_seq) == 0:
            operator.is_leaf_node = True
        else:
            operator.is_leaf_node = False
        self.operator_seq.append(operator)
        self.reversed_ops = self.operator_seq[::-1]
        self.paramters_dict = self.get_parameters()

    # 分类任务设置此项为 ce2 or ce3
    def set_classifier_function(self, classifier_func):
        self.classifier_function = classifier_func

    # 设置损失函数（不应该放在初始化函数中）
    def set_loss_function(self, loss_func):
        self.loss_function = loss_func
    
    # 设置快捷反向传播函数，直接做 a-y, z-y 等等
    # 回归任务设置此项为 mse_loss
    def set_classifier_loss_function(self, combined_func):
        self.classifier_loss_function = combined_func
        if isinstance(combined_func, Classifier.LogisticCrossEntropy):
            self.set_classifier_function(Classifier.Logisitic())   # 二分类交叉熵损失函数
            self.set_loss_function(loss.BCE())   # 二分类交叉熵损失函数
            self.net_type = "BinaryClassifier"
        if isinstance(combined_func, Classifier.SoftmaxCrossEntropy):
            self.set_classifier_function(Classifier.Softmax())   # 多分类交叉熵损失函数
            self.set_loss_function(loss.CrossEntropy3())   # 多分类交叉熵损失函数
            self.net_type = "Classifier"

    def forward(self, X):
        data = X
        for op in self.operator_seq:
            data = op.forward(data)
        if self.classifier_function is not None:
            data = self.classifier_function.forward(data)
        return data

    def predict(self, X):
        data = X
        for op in self.operator_seq:
            data = op.predict(data)
        if self.classifier_function is not None:
            data = self.classifier_function.forward(data)
        return data

    def backward(self, predict, label):
        if self.classifier_loss_function is not None:
            delta = self.classifier_loss_function.backward(predict, label)
        else:
            assert(self.loss_function is not None)
            delta = self.loss_function.backward(predict, label)
        for op in self.reversed_ops:
            delta = op.backward(delta)

    def compute_loss(self, predict, label):
        assert(self.loss_function is not None)
        return self.loss_function(predict, label)

    def save(self, name):
        super().save_parameters(name, self.operator_seq)        

    def load(self, name):
        super().load_parameters(name, self.operator_seq)        

    def compute_loss_accuracy(self, x, label):
        #predict = self.predict(x)  # 为啥用这个不行?
        predict = self.forward(x)
        loss = self.compute_loss(predict, label)
        if self.net_type == "Regression":
            accu = r2(label, loss)
        elif self.net_type == "BinaryClassifier":
            accu = tpn2(predict, label)
        elif self.net_type == "Classifier":
            accu = tpn3(predict, label)
        return loss, accu

    def testing(self, x, label):
        predict = self.predict(x)  # 为啥用这个不行?
        #predict = self.forward(x)
        loss = self.compute_loss(predict, label)
        if self.net_type == "Regression":
            accu = r2(label, loss)
        elif self.net_type == "BinaryClassifier":
            accu = tpn2(predict, label)
        elif self.net_type == "Classifier":
            accu = tpn3(predict, label)
        return loss, accu

    def get_parameters(self):
        param_dict = {}
        unique_id = 0
        for op in self.operator_seq:
            op_name = op.__class__.__name__ + "_" + str(unique_id)
            wb = op.get_parameters()
            if wb is not None:
                param_dict[op_name] = wb
            unique_id += 1
        return param_dict

    def update(self, lr):
        for _, WB in self.paramters_dict.items():
            if isinstance(WB, tuple):
                for wb in WB:
                    wb.Update(lr)
            else: # WeightsBias object
                WB.Update(lr)

    def zero_grad(self):
        for _, WB in self.paramters_dict.items():
            if isinstance(WB, tuple):
                for wb in WB:
                    wb.dW = 0
            else: # WeightsBias object
                WB.dW = 0
