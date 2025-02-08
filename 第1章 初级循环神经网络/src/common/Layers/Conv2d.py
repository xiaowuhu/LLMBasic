import numpy as np
from .Operator import Operator
import common.Optimizers as Optimizers
from .WeightsBias import WeightsBias

# 一维卷积
class Conv2d(Operator):
    def __init__(self, 
                 input_shape, # input_shape = (input_channel, input_height, input_width)
                 output_shape, # kernel_shape = (output_channel, kernel_height, kernel_width)
                 stride=1, padding=0,
                 init_method: str="normal", optimizer: str="SGD"):
        super().__init__()
        assert(len(input_shape) == 3)
        assert(len(output_shape) == 3)
        self.input_channel = input_shape[0]  # input_channel 决定了kernel的个数
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.output_channel = output_shape[0]  # output_channel 决定了filter的个数, 一个filter里包含一个或多个kernel
        self.kernel_height = output_shape[1]
        self.kernel_width = output_shape[2]
        self.stride = stride
        self.padding = padding
        self.output_height = 1 + (self.input_height + 2 * self.padding - self.kernel_height)//self.stride
        self.output_width = 1 + (self.input_width + 2 * self.padding - self.kernel_width)//self.stride
        self.output_shape = (self.output_channel, self.output_height, self.output_width)
        if self.stride > 1:
            # 计算 stride=1 时的 delta_map 的宽度
            self.stride_1_out_height = 1 + (self.input_height + 2 * self.padding - self.kernel_height)
            self.stride_1_out_width = 1 + (self.input_width + 2 * self.padding - self.kernel_width)

        self.WB = Conv2dWeightBias(
            self.input_channel, self.output_channel, self.kernel_height, self.kernel_width,
            init_method, optimizer)

    def get_parameters(self):
        return self.WB

    def forward(self, x):
        return self.forward_im2col(x)
    
    def backward(self, dZ):
        return self.backward_col2im(dZ)

    # 简单的二维卷积操作，传入 z 节省每次创建的开销
    # image is 2d,  weight is 2d, z is 2d
    def _conv2d(self, image, weight, output_h, output_w, z, stride=1, padding=0):
        kernel_h, kernel_w = weight.shape
        for h_out in range(output_h):   # H, 遍历输出的高度（即纵向卷积运算次数）
            start_h = h_out * stride    # 用 stride 步长计算起始点
            for w_out in range(output_w):  # W, 遍历输出的宽度（即横向卷积运算次数）
                start_w = w_out * stride
                data_window = image[        # 获得目标数据窗口
                    start_h:start_h+kernel_h, 
                    start_w:start_w+kernel_w]  
                # 这里本来不需要 + 号，但是考虑到外部调用者需要累加在0,1维上，而且 z 是一个全局缓存
                z[h_out, w_out] += np.sum(data_window * weight)
        return z

    # 四维卷积
    def _conv4d(self, x, weights, bias, input_c, output_c, output_h, output_w, stride):
        batch_num = x.shape[0]
        # 准备好输出的数组，用于存放卷积结果
        z = np.zeros((batch_num, output_c, output_h, output_w))
        for n in range(batch_num):                  # N, 获得单样本
            for c_out in range(output_c):           # C_out 遍历卷积核输出通道
                z[n, c_out] += bias[c_out]          # 在输出通道加上对应的偏置 bias
                for c_in in range(input_c):         # C_in 遍历数据输入通道
                    weight = weights[c_out, c_in]   # 获得二维卷积核                    
                    self._conv2d(x[n, c_in], weight, output_h, output_w, z[n, c_out], stride, 0)
        return z

    # 调用四维卷积
    def forward_simple(self, batch_x):
        self.x = batch_x
        self.batch_size = batch_x.shape[0]
        self.batch_z = self._conv4d(
            batch_x, self.WB.W, self.WB.B, self.input_channel,  # input
            self.output_channel, self.output_height, self.output_width, # output
            self.stride)
        return self.batch_z

    # 用 im2col 的方式实现卷积
    def forward_im2col(self, x):
        self.x = x
        self.batch_size = self.x.shape[0]
        assert(self.x.shape == (self.batch_size, self.input_channel, self.input_height, self.input_width))
        self.col_x = self.im2col(self.x, self.kernel_height, self.kernel_width, self.output_height, self.output_width, self.stride, self.padding)
        self.col_w = self.WB.W.reshape(self.output_channel, -1).T
        self.col_b = self.WB.B.reshape(-1, self.output_channel)
        out1 = np.dot(self.col_x, self.col_w) + self.col_b
        out2 = out1.reshape(self.batch_size, self.output_height, self.output_width, -1)
        self.z = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.z

    def backward_transpose(self, dZ):
        self.backward_dw_transpose(dZ)
        self.backward_dx_transpose(dZ)
    
    def backward_col2im(self, dZ):
        self.backward_dw_col2im(dZ)
        if self.is_leaf_node:
            return None
        else:
            return self.backward_dx_col2im(dZ)

    # 计算 dW 的梯度朴素版
    def backward_dw_simple(self, dZ):
        self.WB.dW *= 0  # 先把缓存清零, 后面要用到这个缓冲区
        for n in range(self.batch_size):                # N, 获得单样本
            for c_out in range(self.output_channel):    # C_out 遍历卷积核输出通道
                weight = dZ[n, c_out]        # 按批量8和输出通道2取出误差值当作卷积核
                for c_in in range(self.input_channel): # 3 取一个通道内的image(RGB之一)
                    image = self.x[n, c_in]  # 取出一个通道的image
                    self._conv2d(image, weight, self.kernel_height, self.kernel_width, self.WB.dW[c_out, c_in])
        self.WB.dW /= self.batch_size
        dB = np.sum(dZ, axis=(0, 2, 3)) / self.batch_size   # (2,)
        self.WB.dB = dB.reshape(self.WB.dB.shape)           # (2,) -> (2,1)
        return self.WB.dW, self.WB.dB

    # 用标准卷积但是需要先变形原始数据
    def backward_dw_transpose(self, dZ):
        X = self.x.transpose(1, 0, 2, 3)    # (C, N, H, W)
        W = dZ.transpose(1, 0, 2, 3)        # (C, N, H, W)
        B = np.zeros((W.shape[0], 1))       # (C, 1)
        # 标准卷积，得到 dW 结果
        dW = self._conv4d(X, W, B, self.batch_size, self.output_channel, self.WB.W.shape[2], self.WB.W.shape[3], 1)
        self.WB.dW = dW.transpose(1, 0, 2, 3)  / self.batch_size    # 转置回来 (N, C, H, W)
        dB = np.sum(dZ, axis=(0, 2, 3)) / self.batch_size           # 按输出通道求和
        self.WB.dB = dB.reshape(self.WB.dB.shape)                   # (2,) -> (2,1)
        return self.WB.dW, self.WB.dB

    # 使用 im2col 的方式实现反向传播, 和线性层一样，但是前后需要转置
    def backward_dw_col2im(self, dZ):
        col_dZ = np.transpose(dZ, axes=(0,2,3,1)).reshape(-1, self.output_channel)
        self.WB.dB = np.sum(col_dZ, axis=0, keepdims=True).T / self.batch_size
        col_dW = np.dot(self.col_x.T, col_dZ) / self.batch_size
        self.WB.dW = col_dW.T.reshape(self.output_channel, self.input_channel, 
                                      self.kernel_height, self.kernel_width)
        return self.WB.dW, self.WB.dB

    # 转置然后用 conv4d 
    def backward_dx_transpose(self, dZ):
        # dZ 补零
        pad_h, pad_w = self.calculate_padding_size( # 计算需要补几个 0
            self.output_height, self.output_width,  # output 做为 input
            self.kernel_height, self.kernel_width,  # kernel 做为 kernel
            self.input_height, self.input_width)    # input 做为 output
        dZ_pad = np.pad(dZ, [(0,0), (0,0), (pad_h,pad_h), (pad_w,pad_w)]) # 补 0
        # W 转置并翻转
        Weights = np.rot90(self.WB.W.transpose(1, 0, 2, 3), 2, axes=(2,3))
        B = np.zeros((Weights.shape[0], 1))         # 没用，置0
        dX = self._conv4d(dZ_pad, Weights, B,       # 4d 卷积
                     self.output_channel, self.input_channel, 
                     self.input_height, self.input_width, 1)
        return dX

    # 用 col2im 的方式实现反向传播
    def backward_dx_col2im(self, dZ):
        col_dZ = np.transpose(dZ, axes=(0,2,3,1)).reshape(-1, self.output_channel)
        col_delta_out = np.dot(col_dZ, self.col_w.T)
        dX = self.col2im(
            col_delta_out, self.x.shape, self.kernel_height, self.kernel_width, 
            self.stride, self.padding, 
            self.output_height, self.output_width)
        return dX


    def im2col(self, input_data, filter_h, filter_w, out_h, out_w, stride=1, pad=0):
        if pad > 0:
            img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        else:
            img = input_data
        col_x = np.zeros((self.batch_size, self.input_channel, filter_h, filter_w, out_h, out_w)).astype(np.float32)
        for h in range(filter_h):
            h_max = h + stride * out_h
            for w in range(filter_w):
                w_max = w + stride * out_w
                col_x[:, :, h, w, :, :] = img[:, :, h:h_max:stride, w:w_max:stride]
            #end for
        #end for
        col_x = np.transpose(col_x, axes=(0, 4, 5, 1, 2, 3)).reshape(self.batch_size * out_h * out_w, -1)
        return col_x

    def col2im(self, col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
        N, C, H, W = input_shape
        tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
        tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)).astype(np.float32)
        for i in range(filter_h):
            i_max = i + stride * out_h
            for j in range(filter_w):
                j_max = j + stride * out_w
                img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
            #end for
        #end for
        return img[:, :, pad:H + pad, pad:W + pad]

    def load(self, name):
        wb_value = super().load_from_txt_file(name)
        self.WB.set_WB_value(wb_value)
    
    def save(self, name):
        wb_value = self.WB.get_WB_value()
        super().save_to_txt_file(name, wb_value)

    def calculate_padding_size(self, input_height, input_width, 
            kernel_height, kernel_width, output_height, output_width, stride=1):
        pad_h = ((output_height - 1) * stride - input_height + kernel_height) // 2
        pad_w = ((output_width - 1) * stride - input_width + kernel_width) // 2
        return (pad_h, pad_w)

    # stride 不为1时，要先对传入的误差矩阵补十字0
    def expand_delta_map(self, dZ):
        if self.stride == 1:
            dZ_stride_1 = dZ
        else:
            # 假设如果stride等于1时，卷积后输出的图片大小应该是多少，然后根据这个尺寸调整delta_z的大小
            dZ_stride_1 = np.zeros((self.batch_num, self.output_channel, self.stride_1_out_height))
            # 把误差值填到当stride=1时的对应位置上
            for bs in range(self.batch_num):
                for oc in range(self.output_channel):
                    for j in range(self.output_height):
                        start = j * self.stride
                        dZ_stride_1[bs, oc, start] = dZ[bs, oc, j]
        return dZ_stride_1


class Conv2dWeightBias(WeightsBias):
    def __init__(self, 
                 input_channel, output_channel,
                 kernel_height, kernel_width,
                 init_method="normal", 
                 optimizer="SGD"):
        self.filter_count = output_channel  # output_channel 决定了filter的个数
        self.kernel_count = input_channel  # input_channel 决定了kernel的个数(每个filter里包含一个或多个kernel)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
          # 在torch 中是这样的，三维
        self.W_shape = (self.filter_count, self.kernel_count, self.kernel_height, self.kernel_width)
        self.B_shape = (self.filter_count, 1)
        self.W, self.B = self.create(self.W_shape, self.B_shape, init_method)
        self.dW = np.zeros(self.W.shape).astype(np.float32)
        self.dB = np.zeros(self.B.shape).astype(np.float32)
        self.opt_W = Optimizers.Optimizer.create_optimizer(optimizer)
        self.opt_B = Optimizers.Optimizer.create_optimizer(optimizer)

    # 前后颠倒存入新变量，要保留原始变量 (w1,w2,w3)->(w3,w2,w1)
    def get_flip(self):
        self.WT = np.zeros_like(self.W)
        for i in range(self.filter_count):
            for j in range(self.kernel_count):
                self.WT[i,j] = np.flip(self.W[i,j])
        return self.WT

    def create(self, w_shape, b_shape, method):
        assert(len(w_shape) == 4)
        num_input = w_shape[3]
        num_output = 1
        
        if method == "zero":
            W = np.zeros(w_shape).astype(np.float32)
        elif method == "normal":
            W = np.random.normal(0, 1, w_shape).astype(np.float32)
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_input*num_output), w_shape).astype(np.float32)
        elif method == "xavier":
            t = np.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, w_shape).astype(np.float32)
        
        B = np.zeros(b_shape).astype(np.float32)
        return W, B

    # 因为w,b 的形状不一样，需要reshape成 n行1列的，便于保存
    def get_WB_value(self):
        value = np.concatenate((self.W.reshape(-1,1), self.B.reshape(-1,1)))
        return value

    def get_dWB_value(self):
        value = np.concatenate((self.dW.reshape(-1,1), self.dB.reshape(-1,1)))
        return value

    # 先把一列数据读入，然后变形成为对应的W,B的形状
    def set_WB_value(self, value):
        self.W = value[0:-self.filter_count].reshape(self.W_shape)
        self.B = value[-self.filter_count:].reshape(self.B_shape)

    def set_dWB_value(self, value):        
        self.dW = value[0:-self.filter_count].reshape(self.W_shape)
        self.dB = value[-self.filter_count:].reshape(self.B_shape)
