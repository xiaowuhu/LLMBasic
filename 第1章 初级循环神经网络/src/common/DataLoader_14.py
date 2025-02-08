import os
import numpy as np
import struct

class DataLoader_14(object):
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    # 加载数据, 只加载指定的列号，最后一列是标签值
    def load_txt_data(self, col_list: list = None):
        # load train data
        self.train_data_raw = np.loadtxt(self.train_file_path)
        if col_list is not None:
            self.num_feature = len(col_list) - 1
            self.tmp_data = np.zeros((self.train_data_raw.shape[0], len(col_list)))
            for i, col in enumerate(col_list):
                self.tmp_data[:, i] = self.train_data_raw[:, col]
            self.train_data_raw = self.tmp_data
        else:
            self.num_feature = self.train_data_raw.shape[1] - 1
        self.train_x = self.train_data_raw[:,0:self.num_feature]
        self.train_y = self.train_data_raw[:,-1:]
        self.num_sample = self.train_x.shape[0]
        self.num_train = self.num_sample
        self.num_val = 0
        self.num_feature = self.train_x.shape[1]
        # load test data
        self.test_data_raw = np.loadtxt(self.test_file_path)
        self.test_x = self.test_data_raw[:,0:self.num_feature]
        self.test_y = self.test_data_raw[:,-1:]
    
    # mode could be "image"(1,28,28) or "vector"(1,784)
    def load_npz_data(self, mode="image"):
        npz_data = np.load(self.train_file_path)
        self.train_x = npz_data["data"].astype(np.float32)
        self.train_y = npz_data["label"].astype(np.float32)
        self.num_sample = self.train_x.shape[0]
        self.num_feature = self.train_x.shape[1]
        self.num_train = self.num_sample
        self.num_val = 0
        self.num_classes = len(np.unique(self.train_y))
        # load test data
        npz_data = np.load(self.test_file_path)
        self.test_x = npz_data["data"].astype(np.float32)
        self.test_y = npz_data["label"].astype(np.float32)
        self.num_test = self.test_x.shape[0]
        # mode
        if mode == "vector":
            self.train_x = np.reshape(self.train_x, (self.num_train, -1))
            self.test_x = np.reshape(self.test_x, (self.num_test, -1))
            self.num_feature = self.train_x.shape[1]


    def add_channel_info(self):
        self.train_x = np.expand_dims(self.train_x, axis=1)
        self.test_x = np.expand_dims(self.test_x, axis=1)
        self.num_channel = 1

    def load_FashionMNIST_data(self, mode="image"):
        return self.load_MNIST_data(mode=mode)

    # 调用前先手动把四个文件放到当前目录下
    def load_MNIST_data(self, mode="image"):
        train_image_file = os.path.join(self.train_file_path, 'train-images-idx3-ubyte')
        train_label_file = os.path.join(self.train_file_path, 'train-labels-idx1-ubyte')
        test_image_file = os.path.join(self.test_file_path, 't10k-images-idx3-ubyte')
        test_label_file = os.path.join(self.test_file_path, 't10k-labels-idx1-ubyte')
        self.train_x = self.read_MNIST_image_file(train_image_file)
        self.train_y = self.read_MINIST_label_file(train_label_file)
        self.test_x = self.read_MNIST_image_file(test_image_file)
        self.test_y = self.read_MINIST_label_file(test_label_file)
        if mode == "image":
            self.num_sample = self.train_x.shape[0]
            self.num_train = self.num_sample
            self.num_val = 0
            self.num_feature = self.train_x.shape[1]
            self.num_test = self.test_x.shape[0]
        elif mode == "vector":
            self.num_sample = self.train_x.shape[0]
            self.num_train = self.num_sample
            self.num_test = self.test_x.shape[0]
            self.num_val = 0
            self.train_x = np.reshape(self.train_x, (self.num_train, -1))
            self.test_x = np.reshape(self.test_x, (self.num_test, -1))
            self.num_feature = self.train_x.shape[1]

    def load_EMNIST_data(self, mode="image"):
        train_image_file = os.path.join(self.train_file_path, 'emnist-letters-train-images-idx3-ubyte')
        train_label_file = os.path.join(self.train_file_path, 'emnist-letters-train-labels-idx1-ubyte')
        test_image_file = os.path.join(self.test_file_path, 'emnist-letters-test-images-idx3-ubyte')
        test_label_file = os.path.join(self.test_file_path, 'emnist-letters-test-labels-idx1-ubyte')
        self.train_x = self.read_MNIST_image_file(train_image_file)
        self.train_y = self.read_MINIST_label_file(train_label_file) - 1 # 变成 0-25
        self.test_x = self.read_MNIST_image_file(test_image_file)
        self.test_y = self.read_MINIST_label_file(test_label_file) - 1 # 变成 0-25
        self.num_sample = self.train_x.shape[0]
        self.num_train = self.num_sample
        self.num_val = 0
        self.num_feature = self.train_x.shape[1]
        self.num_test = self.test_x.shape[0]


    # output array: num_images * channel * 28 * 28
    # due to gray image instead of color, so channel = 1
    def read_MNIST_image_file(self, image_file_name):
        # header
        f = open(image_file_name, "rb")
        a = f.read(4)
        b = f.read(4)
        num_images = int.from_bytes(b, byteorder='big')
        c = f.read(4)
        num_rows = int.from_bytes(c, byteorder='big')
        d = f.read(4)
        num_cols = int.from_bytes(d, byteorder='big')
        # image data binary
        image_size = num_rows * num_cols    # 28x28=784
        fmt = '>' + str(image_size) + 'B'
        image_data = np.empty((num_images,1,num_rows,num_cols)) # N x 1 x 28 x 28
        for i in range(num_images):
            bin_data = f.read(image_size)   # read 784 byte data for one time
            unpacked_data = struct.unpack(fmt, bin_data)
            array_data = np.array(unpacked_data)
            array_data2 = array_data.reshape((1, num_rows, num_cols))
            image_data[i] = array_data2
        # end for
        f.close()
        return image_data.astype(np.float32)

    def read_MINIST_label_file(self, lable_file_name):
        f = open(lable_file_name, "rb")
        f.read(4)
        a = f.read(4)
        num_labels = int.from_bytes(a, byteorder='big')

        fmt = '>B'
        label_data = np.zeros((num_labels,1))   # N x 1
        for i in range(num_labels):
            bin_data = f.read(1)
            unpacked_data = struct.unpack(fmt, bin_data)[0]
            label_data[i] = unpacked_data
        f.close()
        return label_data.astype(np.int32)


    # 分出验证集（但是需要先做归一化,然后打乱数据,再分验证集）
    def split_data(self, ratio=0.8):
        self.num_train = int(self.num_sample * ratio)
        # 验证集
        self.num_val = self.num_sample - self.num_train
        self.val_x = self.train_x[self.num_train:, :]
        self.val_y = self.train_y[self.num_train:, :]
        # 训练集
        self.train_x = self.train_x[:self.num_train, :]
        self.train_y = self.train_y[:self.num_train, :]

    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.num_train)
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]

    # =============================================

    # 0,1归一化训练数据
    def MinMaxScaler_X(self, is_image=False):
        if is_image:
            self.x_min = np.min(self.train_x)
            self.x_max = np.max(self.train_x)
        else:
            self.x_min = np.min(self.train_x, axis=0)
            self.x_max = np.max(self.train_x, axis=0)
        self.train_x = (self.train_x - self.x_min) / (self.x_max - self.x_min)

    # 0,1 归一化预测数据x
    def MinMaxScaler_pred_X(self, X):
        normalized_x = (X - self.x_min) / (self.x_max - self.x_min)
        return normalized_x

    # 0,1反归一化预测数据
    def de_MinMaxScaler_X(self, X):
        de_normalized_X = X * (self.x_max - self.x_min) + self.x_min
        return de_normalized_X

    # =============================================
    
    # 0,1归一化标签数据
    def MinMaxScaler_Y(self):
        self.y_min = np.min(self.train_y, axis=0)
        self.y_max = np.max(self.train_y, axis=0)
        self.train_y = (self.train_y - self.y_min) / (self.y_max - self.y_min)

    # 0,1 归一化预测数据y
    def MinMaxScaler_pred_Y(self, Y):
        normalized_y = (Y - self.y_min) / (self.y_max - self.y_min)
        return normalized_y

    # 0,1 反归一化预测结果
    def de_MinMaxScaler_Y(self, pred_Y):
        de_normalized_Y = pred_Y * (self.y_max - self.y_min) + self.y_min
        return de_normalized_Y

    # =============================================

    # 0 标准化训练数据
    def StandardScaler_X(self, is_image=False, mean:float=None, std:float=None):
        if is_image:  # 整体归一化，因为有很多 0
            if mean is not None:
                self.x_mean = mean
            else:
                self.x_mean = np.mean(self.train_x)
            if std is not None:
                self.x_std = std
            else:
                self.x_std = np.std(self.train_x)
        else: # 按列归一化
            self.x_mean = np.mean(self.train_x, axis=0)
            self.x_std = np.std(self.train_x, axis=0)
        self.train_x = (self.train_x - self.x_mean) / self.x_std

    # 0 标准化预测数据
    def StandardScaler_pred_X(self, X):
        normalized_x = (X - self.x_mean) / self.x_std
        return normalized_x

    # 0 反标准化预测数据
    def de_StandardScaler_X(self, X):
        de_normalized_X = X * self.x_std + self.x_mean
        return de_normalized_X

    # =============================================

    # 0 标准化标签数据
    def StandardScaler_Y(self):
        self.y_mean = np.mean(self.train_y, axis=0)
        self.y_std = np.std(self.train_y, axis=0)
        self.train_y = (self.train_y - self.y_mean) / self.y_std

    # 0 反标准化预测结果
    def de_StandardScaler_Y(self, pred_Y):
        de_normalized_Y = pred_Y * self.y_std + self.y_mean
        return de_normalized_Y

    # =============================================

    # 变成Onehot编码，在split_data之前调用本方法
    def to_onehot(self, num_classes):
        self.num_classes = num_classes
        self.train_y = np.eye(self.num_classes)[self.train_y.flatten().astype(np.int32)].astype(np.float32)
        if hasattr(self, 'test_y'):
            self.test_y = np.eye(self.num_classes)[self.test_y.flatten().astype(np.int32)].astype(np.float32)

    # get batch training data
    def get_batch(self, batch_size, batch_id):
        start = batch_id * batch_size
        end = start + batch_size
        batch_X = self.train_x[start:end]
        batch_Y = self.train_y[start:end]
        return batch_X, batch_Y

    def get_val(self, num: int=-1):
        if num == -1:  # full size
            return self.val_x, self.val_y
        else:
            indices = np.random.choice(self.num_val, num)
            return self.val_x[indices], self.val_y[indices]

    def get_train(self):
        return self.train_x, self.train_y

    def get_test(self):
        return self.test_x, self.test_y

    def get_batch_test(self, batch_size, batch_id):
        start = batch_id * batch_size
        end = start + batch_size
        batch_X = self.test_x[start:end]
        batch_Y = self.test_y[start:end]
        return batch_X, batch_Y
