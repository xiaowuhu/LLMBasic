import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def get_cosine_scaled_reward(
    gen_len,
    min_value:float=0.5,  # 正确答案最小值
    max_value:float=1.0,  # 正确答案最大值
    max_len=1000
):
    progress = gen_len / max_len
    cosine = np.cos(progress * np.pi) # Cosine value based on progress
    reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
    return reward

if __name__=="__main__":
    t = np.arange(0, 900)
    r = get_cosine_scaled_reward(t) #,-0.5,-0.1)
    plt.plot(t,r)
    plt.grid()
    plt.xlabel("输出长度")
    plt.ylabel("分数")
    plt.show()
