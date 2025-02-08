import numpy as np
from .Operator import Operator
from .WeightsBias import WeightsBias


# 一维卷积
class Conv1d(Operator):
    def __init__(self, 
                 input_shape, # input_shape = (input_channel, input_length)
                 output_shape, # kernal_shape = (output_channel, kernal_length)
                 stride=1, padding=0,
                 init_method: str="normal", optimizer: str="SGD"):
        assert(len(input_shape) == 2)
        assert(len(output_shape) == 2)
        self.input_channel = input_shape[0]  # input_channel 决定了kernal的个数
        self.input_length = input_shape[1]
        self.kernel_length = output_shape[1]
        self.output_channel = output_shape[0]  # output_channel 决定了filter的个数, 一个filter里包含一个或多个kernal
        self.stride = stride
        self.padding = padding
        self.output_length = 1 + (self.input_length + 2 * self.padding - self.kernel_length)//self.stride
        if self.stride > 1:
            # 计算 stride=1 时的 delta_map 的宽度
            self.stride_1_out_len = 1 + (self.input_length + 2 * self.padding - self.kernel_length) // 1

        self.WB = Conv1dFilters(
            self.input_channel, self.output_channel, self.kernel_length, 
            init_method, optimizer)

    def get_parameters(self):
        return self.WB

    def forward(self, batch_x):
        self.input_data = batch_x
        self.m = batch_x.shape[0]
        self.batch_output = np.zeros((self.m, self.output_channel, self.output_length))
        for i in range(self.m):  # N
            row_data = self.input_data[i]  # 获得单个样本数据
            for oc in range(self.output_channel):  # 遍历输出通道
                for ic in range(self.input_channel):  # 遍历输入通道
                    kernal = self.WB.W[oc, ic]  # 根据输出、输入通道获得对应的卷积核
                    for j in range(self.output_length):  # 遍历输出的长度（即卷积运算次数）
                        start = j * self.stride  # 用步长计算起始点，步长为 2 时得到起点序列：0,2,4...
                        data_window = row_data[ic, start:start+self.kernel_length]  # 获得目标数据窗口
                        # 计算卷积并把多个 kernal 在 row_data[ic] 上的计算结果合并
                        self.batch_output[i, oc, j] += np.sum(np.multiply(data_window, kernal))
        return self.batch_output

    def backward(self, delta_in):
        if self.stride > 1:
            delta_in_stride = self.expand_delta_map(delta_in)
        else:
            delta_in_stride = delta_in

        # 求权重梯度, 用 row_data 和 kernal 做卷积
        dW = np.zeros_like(self.WB.dW)
        for i in range(self.m):
            row_data = self.input_data[i]
            filter = delta_in_stride[i]
            assert(self.WB.dW.shape[0] == filter.shape[0])

            out_c = self.WB.W.shape[0] # 3
            in_c = self.input_channel  # 2
            out_len = self.WB.W.shape[2] # 4
            k_len = delta_in_stride.shape[2]  # 2
            stride = 1

            for oc in range(out_c):
                # 用delta_in当作filter,返回 NxCxW, N是批量
                # C为网络输出通道数，在这里当作输入通道数
                kernal = filter[oc]  
                for ic in range(in_c):
                    for j in range(out_len):
                        jj = j * stride
                        data_window = row_data[ic, jj:jj+k_len]
                        dW[oc, ic, j] += np.sum(np.multiply(data_window, kernal))

        self.WB.dW = dW / self.m
        #self.WB.dB = np.sum(delta_in, keepdims=True) / self.m
        #print(self.WB.dW * self.m)


        # 求输出梯度
        # pad_left,pad_right = self.calculate_padding_size(
        #     delta_in_stride.shape[1], self.kernel_length, self.input_size)
        # delta_in_pad = np.pad(delta_in_stride,((0,0),(pad_left,pad_right)))
        # delta_out = np.zeros(self.input_data.shape)
        # kernal = np.flip(self.WB.W)
        # for i in range(self.m):
        #     row_data = delta_in_pad[i:i+1]
        #     delta_out[i] = self._conv1d_single_row_data(row_data, kernal)
        # return delta_out

    def load(self, name):
        wb_value = super().load_from_txt_file(name)
        self.WB.set_value(wb_value)
    
    def save(self, name):
        wb_value = self.WB.get_value()
        super().save_to_txt_file(name, wb_value)

    def calculate_padding_size(self, input_size, kernal_size, output_size, stride=1):
        pad_w = ((output_size - 1) * stride - input_size + kernal_size) // 2
        return (pad_w, pad_w)

    # stride 不为1时，要先对传入的误差矩阵补十字0
    def expand_delta_map(self, dZ):
        if self.stride == 1:
            dZ_stride_1 = dZ
        else:
            # 假设如果stride等于1时，卷积后输出的图片大小应该是多少，然后根据这个尺寸调整delta_z的大小
            dZ_stride_1 = np.zeros((self.m, self.output_channel, self.stride_1_out_len))
            # 把误差值填到当stride=1时的对应位置上
            for bs in range(self.m):
                for oc in range(self.output_channel):
                    for j in range(self.output_length):
                        start = j * self.stride
                        dZ_stride_1[bs, oc, start] = dZ[bs, oc, j]
        return dZ_stride_1


class Conv1dFilters(object):
    def __init__(self, 
                 input_channel,
                 output_channel,
                 kernal_length,  # 一维卷积只有长度，没有宽度（宽度=1）
                 init_method="normal", 
                 optimizer="SGD"):
        self.filter_count = output_channel  # output_channel 决定了filter的个数
        self.kernal_count = input_channel  # input_channel 决定了kernal的个数(每个filter里包含一个或多个kernal)
        self.kernal_length = kernal_length
          # 在torch 中是这样的，三维
        self.W_shape = (self.filter_count, self.kernal_count, self.kernal_length)
        self.B_shape = (self.filter_count, 1)
        self.W, self.B = self.create(self.W_shape, self.B_shape, init_method)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    # 前后颠倒存入新变量，要保留原始变量 (w1,w2,w3)->(w3,w2,w1)
    def get_flip(self):
        self.WT = np.zeros_like(self.W)
        for i in range(self.filter_count):
            for j in range(self.kernal_count):
                self.WT[i,j] = np.flip(self.W[i,j])
        return self.WT

    def create(self, w_shape, b_shape, method):
        assert(len(w_shape) == 3)
        num_input = w_shape[2]
        num_output = 1
        
        if method == "zero":
            W = np.zeros(w_shape)
        elif method == "normal":
            W = np.random.normal(0, 1, w_shape)
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_input*num_output), w_shape)
        elif method == "xavier":
            t = np.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, w_shape)
        
        B = np.zeros(b_shape)
        return W, B

    def Update(self, lr):
        self.W -= lr * self.dW
        self.B -= lr * self.dB

    def get_value(self):
        # 因为w,b 的形状不一样，需要reshape成 n行1列的，便于保存
        value = np.concatenate((self.W.reshape(-1,1), self.B.reshape(-1,1)))
        return value

    def set_value(self, value):
        assert (value.ndim == 1)
        self.W = value[0:-self.filter_count].reshape(self.W_shape)
        self.B = value[-self.filter_count:].reshape(self.B_shape)
