import numpy as np
from .Operator import Operator


# 一维池化
class Pool2d(Operator):
    def __init__(self, 
                 input_shape,  # (input_channel, input_h, input_w)
                 kernel_shape,  # (pool_height, pool_width)
                 stride = 1,
                 padding = 0,
                 pool_type = "max", # or "mean"/"average"
    ):
        self.input_channel = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.kernel_height = kernel_shape[0]
        self.kernel_width = kernel_shape[1]
        self.kernel_size = self.kernel_height * self.kernel_width
        self.stride = stride
        self.padding = padding
        assert(padding == 0)  # 暂不处理非0的情况
        self.pool_type = pool_type
        assert(pool_type == "max")  # 暂不处理 average 
        self.output_height = 1 + (self.input_height + 2 * self.padding - self.kernel_height) // self.stride
        self.output_width = 1 + (self.input_width + 2 * self.padding - self.kernel_width) // self.stride
        self.output_shape = (self.input_channel, self.output_height, self.output_width)
    
    def forward(self, x):
        return self.forward_im2col(x)
    
    def backward(self, dZ):
        return self.backward_col2im(dZ)

    def forward_simple(self, x):
        self.x = x
        assert(self.input_height == self.x.shape[2])
        assert(self.input_width == self.x.shape[3])
        assert(self.input_channel == self.x.shape[1])
        self.batch_size = self.x.shape[0]
        self.z = np.zeros((self.batch_size, self.input_channel, self.output_height, self.output_width)).astype(np.float32)
        self.argmax = np.zeros_like(self.z).astype(np.int32)

        for n in range(self.batch_size):
            for c_in in range(self.input_channel):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        start_h = h * self.stride
                        start_w = w * self.stride
                        end_h = start_h + self.kernel_height
                        end_w = start_w + self.kernel_width
                        data_window = self.x[n, c_in, start_h:end_h, start_w:end_w]
                        self.z[n, c_in, h, w] = np.max(data_window)
                        self.argmax[n, c_in, h, w] = np.argmax(data_window)
        return self.z

    def backward_simple(self, dz):
        dx = np.zeros_like(self.x)
        for n in range(self.batch_size):
            for c_in in range(self.input_channel):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        pos = self.argmax[n, c_in, h, w]
                        i, j = np.unravel_index(pos, (self.kernel_height, self.kernel_width))
                        dx[n, c_in, h * self.stride + i, w * self.stride + j] = dz[n, c_in, h, w]
        return dx

    def forward_im2col(self, x):
        self.x = x
        self.batch_size = self.x.shape[0]
        col = self.im2col(x, self.kernel_height, self.kernel_width, 
                          self.output_height, self.output_width, self.stride, 0)
        col_x = col.reshape(-1, self.kernel_height * self.kernel_width)
        self.arg_max = np.argmax(col_x, axis=1) # 取一行中的最大值的位置
        out1 = np.max(col_x, axis=1)  # 取一行中的最大值
        out2 = out1.reshape(self.batch_size, self.output_height, self.output_width, self.input_channel)
        self.z = np.transpose(out2, axes=(0,3,1,2))
        return self.z

    def backward_col2im(self, dZ):
        dout = np.transpose(dZ, (0,2,3,1))
        dmax = np.zeros((dout.size, self.kernel_size)).astype(np.float32)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (self.kernel_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = self.col2im(dcol, self.x.shape, self.kernel_height, self.kernel_width, self.stride, 0, self.output_height, self.output_width)
        return dx

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
        tmp1 = col.reshape(self.batch_size, out_h, out_w, C, filter_h, filter_w)
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
