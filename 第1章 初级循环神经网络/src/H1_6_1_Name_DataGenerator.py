# 读取 src/data/names/*.txt files
# 变成unicode
# 按长度分组，便于后面的 batch training
# name - category 
# create one-hot tensor for the char in name, each char is one timestep
# 每个字符是 54 个特征，one hot,26x2+2 (空格和')

import os
import glob
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " '-"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

def nameToTensor(name):
    tensor = torch.zeros(len(name), n_letters)
    for li, letter in enumerate(name):
        tensor[li][letterToIndex(letter)] = 1
    return tensor

# 把名字变成tensor保存起来
def change_names_to_tensor():
    print("reading data...")
    file_path = "../data/names/*.txt"
    allfiles = glob.glob(file_path)
    all_categories = []
    all_X = [None] * 30  # 记录以长度为 key 的名字 group（名字，类别）,预估最长名字不超过30个字符
    for category_id, filename in enumerate(allfiles):
        category_name = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category_name)
        unicode_names = open(filename, encoding='utf-8').read().strip().split('\n')
        set_unicode_names = set(unicode_names)  # 去重
        for name in set_unicode_names:
            ascii_name = unicodeToAscii(name)
            len_of_name = len(ascii_name)
            if all_X[len_of_name] is None:
                all_X[len_of_name] = []  # 添加一个 list
            # 向 list 中添加 (name, id)
            all_X[len_of_name].append((ascii_name, category_id))

    X = []
    Y = []

    for len_name, name_country_tuple in enumerate(all_X):
        if name_country_tuple is not None:
            count = len(name_country_tuple)
            # 具有相同长度的名字及其国家编号
            X_len = torch.zeros(count, len_name, n_letters)  
            Y_len = torch.zeros(count, dtype=torch.long)
            X.append(X_len)
            Y.append(Y_len)
            for i, x in enumerate(name_country_tuple):
                X_len[i] = nameToTensor(x[0])
                Y_len[i] = x[1]
    print("done")
    # for i in range(len(X)):
    #     print(X[i].shape, Y[i].shape)
    return X, Y, all_categories
    

class DataLoader_train(object):
    def __init__(self, num_class, batch_size):
        self.num_class = num_class
        self.batch_size = batch_size
        self.group_id = 0
        self.pos_id_in_group = 0
        self.X, self.Y, self.category_name = change_names_to_tensor()
        self.group_count = len(self.X)
    
    def shuffle(self):
        for i in range(self.group_count):
            idx = torch.randperm(self.X[i].shape[0])
            self.X[i] = self.X[i][idx]
            self.Y[i] = self.Y[i][idx]

    def get_batch(self):
        # 从 X 中按顺序取数据,需要记住上一次取的位置
        start = self.pos_id_in_group
        end = start + self.batch_size
        batch_x = self.X[self.group_id][start:end]
        batch_y = self.Y[self.group_id][start:end]
        self.pos_id_in_group = end
        if end >= self.X[self.group_id].shape[0]: # 本组样本已经越界
            self.pos_id_in_group = 0  # 归零
            self.group_id += 1  # 下次取下一组
            if self.group_id >= self.num_class:  # 组数越界
                self.group_id = 0  # 归零，从头开始
        return batch_x, batch_y
    
    def is_epoch_done(self):
        if self.group_id == 0 and self.pos_id_in_group == 0:
            return True
        else:
            return False

    def get_test_data(self):
        return self.X[4], self.Y[4]  # 只用名字长度为 5 的数据进行验证

    def get_category_name(self):
        return self.category_name

if __name__=="__main__":
    data_loader = DataLoader_train(18, 32)
    # for epoch in range(2):
    #     print("===== epoch", epoch)
    #     data_loader.shuffle()
    #     epoch_done = False
    #     while epoch_done == False:
    #         x, y = data_loader.get_batch(32)
    #         print(x.shape, y.shape)
    #         epoch_done = data_loader.is_epoch_done()
