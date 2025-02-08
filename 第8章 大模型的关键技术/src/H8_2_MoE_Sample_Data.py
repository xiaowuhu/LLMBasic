
# 向原作者 Shahriar Hossain 致敬
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def draw_sample(x, y, num_sample):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(3):
        j = i * num_sample // 3
        if y[j] == 0:
            c = 'r'
            m = 'o'
        elif y[j] == 1:
            c = 'g'
            m = '*'
        else:
            c = 'b'
            m = '^'
        ax.scatter(x[j:j+10,0], x[j:j+10,1], x[j:j+10,2], c=c, marker=m, label=str(i))
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'$x_2$')
    ax.legend()
    plt.show()


def generate_data(num_samples, input_dim):
    np.random.seed(5)

    # 生成标签值，分别为 0, 1, 2
    y_data=torch.cat([
        torch.zeros(num_samples//3), # 0
        torch.ones(num_samples//3),  # 1
        torch.full((num_samples - 2 * (num_samples//3),), 2)  # 2
    ]).long()

    # 生成样本值，但需要后续调整
    x_data=torch.randn(num_samples, input_dim)

    # 调整
    for i in range(num_samples):
        if y_data[i] == 0:  # 第0类样本
            x_data[i, 0] += 1  # Making x[0] more positive
        elif y_data[i] == 1:  # 第1类样本
            x_data[i, 1] -= 1  # Making x[1] more negative
        elif y_data[i] == 2:  # 第2类样本
            x_data[i, 0] -= 1  # Making x[0] more negative

    # 绘图
    #draw_sample(x_data, y_data, num_samples)

    # 打乱顺序
    indices=torch.randperm(num_samples)
    x_data=x_data[indices]
    y_data=y_data[indices]


    # Verify the label distribution
    print("三类样本分布:", y_data.bincount())

    # Splitting data for training individual experts
    # 用前一半数据训练专家
    x_train_experts=x_data[:int(num_samples/2)]
    y_train_experts=y_data[:int(num_samples/2)]
    # 专家 1 只负责类别 0 和 1
    mask_expert1 = (y_train_experts == 0) | (y_train_experts == 1)
    # 专家 2 只负责类别 1 和 2
    mask_expert2 = (y_train_experts == 1) | (y_train_experts == 2)
    # 专家 3 只负责类别 0 和 2
    mask_expert3 = (y_train_experts == 0) | (y_train_experts == 2)

    # 取三类样本的最小数量
    num_samples_per_expert = min(mask_expert1.sum(), mask_expert2.sum(), mask_expert3.sum())
    # 得到专家 1 的训练样本
    x_train_expert1 = x_train_experts[mask_expert1][:num_samples_per_expert]
    y_train_expert1 = y_train_experts[mask_expert1][:num_samples_per_expert]
    # 得到专家 2 的训练样本
    x_train_expert2 = x_train_experts[mask_expert2][:num_samples_per_expert]
    y_train_expert2 = y_train_experts[mask_expert2][:num_samples_per_expert]
    # 得到专家 3 的训练样本
    x_train_expert3 = x_train_experts[mask_expert3][:num_samples_per_expert]
    y_train_expert3 = y_train_experts[mask_expert3][:num_samples_per_expert]
    print("训练专家数据:")
    print(f'专家1: {x_train_expert1.shape}, 专家2:{x_train_expert2.shape}, 专家3:{x_train_expert3.shape}')

    # 用后一半数据训练 MoE
    x_remaining=x_data[int(num_samples/2):]
    y_remaining=y_data[int(num_samples/2):]
    # 按 0.8 : 0.2 分割训练集和测试集（没有验证集）
    split=int(0.8*len(x_remaining))
    # 训练集
    x_train_moe=x_remaining[:split]
    y_train_moe=y_remaining[:split]
    # 测试集
    x_test_moe=x_remaining[split:]
    y_test_moe=y_remaining[split:]
    print("训练 MoE 数据:")
    print(f'训练数据: {x_train_moe.shape}, 测试数据:{x_test_moe.shape}')

    train_expert = [(x_train_expert1, y_train_expert1), (x_train_expert2, y_train_expert2), (x_train_expert3, y_train_expert3)]
    train_moe = (x_train_moe, y_train_moe)
    test_moe = (x_test_moe, y_test_moe)
    return train_expert, train_moe, test_moe

if __name__=="__main__":
    generate_data(6000, 3)
