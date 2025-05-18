## FP8中防止精度下溢伪代码
def MMA_Mixed_Precision(A_fp8, W_fp8, scal_A, scal_W, NC=128):
    """
    使用混合精度（FP8 + FP32）进行矩阵乘法累加（MMA）操作。

    参数：
    - A_fp8: 形状为 [M, K] 的 FP8 激活值矩阵。
    - W_fp8: 形状为 [K, N] 的 FP8 权重矩阵。
    - scal_A: A_fp8 的量化缩放因子，形状为 [M, K // NC]。
    - scal_W: W_fp8 的量化缩放因子，形状为 [K // NC, N]。
    - NC: 量化分组大小，默认为 128。

    返回：
    - final_answer: 高精度的 FP32 结果矩阵，形状为 [M, N]。
    """
    import numpy as np

    # 获取输入矩阵的形状
    M, K = A_fp8.shape
    K, N = W_fp8.shape

    # 初始化最终结果矩阵
    final_answer = np.zeros((M, N), dtype=np.float32)

    # 遍历 A 的行
    for i in range(M):  # 遍历 A 的每一行
        # 遍历 W 的列，以 NC 为步长
        for j in range(0, N, NC):  # 遍历 W 的每一列，每次处理 NC 列
            # 初始化累加器，形状为 [1, NC]
            D_fp32 = np.zeros((1, NC), dtype=np.float32)

            # 遍历内部维度 K，以 NC 为步长
            for k in range(0, K, NC):
                # 提取 A 的当前块 [1, NC]
                A_block = A_fp8[i, k:k+NC]  # 形状为 [1, NC]

                # 提取 W 的当前块 [NC, NC]
                W_block = W_fp8[k:k+NC, j:j+NC]  # 形状为 [NC, NC]

                # 在 Tensor 核上使用 FP8 计算矩阵乘法
                C_fp8 = np.dot(A_block, W_block)  # 形状为 [1, NC]

                # 提取对应的缩放因子
                scale_A = scal_A[i, k // NC]  # A 的缩放因子
                scale_W = scal_W[k // NC, j:j+NC]  # W 的缩放因子，形状为 [1, NC]

                # 在 CUDA 核中进行反量化
                D_fp32 += scale_A * scale_W * C_fp8  # 累加到 D_fp32

            # 将累加结果存储到最终答案矩阵
            final_answer[i, j:j+NC] = D_fp32

    # 返回最终结果
    return final_answer
