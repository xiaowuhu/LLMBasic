import torch
import numpy as np

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

    for i in range(num_samples):
        if y_data[i] == 0:
            tmp = x_data[i].tolist()
            tmp.sort()
            x_data[i] = torch.tensor(tmp)
        elif y_data[i] == 1:
            tmp = x_data[i].tolist()
            tmp.sort()
            x_data[i] = torch.tensor(tmp[::-1])
        elif y_data[i] == 2:
            tmp = x_data[i].tolist()
            tmp.sort()
            for j in range(0, len(tmp)//2):
                x_data[i, j] = tmp[j]  # 当前最小的放在前面
                x_data[i, len(tmp)-j-1] = tmp[j+1] # 当前第二小的放在后面

    # 绘图
    #draw_sample(x_data, y_data, num_samples)

    # 打乱顺序
    indices=torch.randperm(num_samples)
    x_data=x_data[indices]
    y_data=y_data[indices]

    # Verify the label distribution
    print("三类样本分布:", y_data.bincount())

    split=int(0.8*len(x_data))
    # Splitting data for training individual experts
    # 用前一半数据训练专家
    x_train = x_data[0:split]
    y_train = y_data[0:split]
    x_test = x_data[split:]
    y_test = y_data[split:]

    return x_train, y_train, x_test, y_test

if __name__=="__main__":
    x_train, y_train, x_test, y_test = generate_data(6000, 10)
    print(f"train: {x_train.shape}, test: {x_test.shape}")
