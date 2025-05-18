import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def rotate(theta, x):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    r = np.dot(x, R)
    return r

def draw1():
    degree = -15
    theta = degree / 180 * np.pi
    x = np.array([0.966, 0.259])
    r = rotate(theta, x)
    plt.plot((0,x[0]), (0,x[1]))
    plt.plot((0,r[0]),(0,r[1]))
    plt.axis('equal')
    plt.grid()
    plt.show()
    print(f"{x} -> {r}, theta={degree}")


def create_sin_cos_table(L, d):
    j = np.arange(0, d/2, 1)  # j=[0,1,...,d/2-1]
    theta = np.power(10000, -2*j/d)
    #theta = 10000 ** (-np.arange(0, d, 2) / d)
    theta = theta.repeat(2,axis=0).reshape(1,-1)
    pos = np.arange(0, L).reshape(-1,1) # pos=[0,1,2,...]
    table = np.dot(pos, theta)
    sin_table = np.sin(table)
    sin_table[:,1::2] = -sin_table[:,1::2]
    cos_table = np.cos(table)
    return sin_table, cos_table

# change [0,1,2,3,4,5] -> [1,0,3,2,5,4]
def swich_neighbor(vec):
    return vec.reshape(-1, 2)[:, ::-1].flatten()

# 旋转
def rotary(x, pos, cos_table, sin_table):
    x2 = swich_neighbor(x)
    return x * cos_table[pos] + x2 * sin_table[pos]

def create_sin_cos_cache2(max_num_tokens, d):
    theta = 10000 ** (-np.arange(0, d, 2) / d)
    theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()

    pos = np.arange(0, max_num_tokens)
    table = pos.reshape(-1, 1) @ theta.reshape(1, -1)  # [max_num_tokens, head_size]

    sin_cache = np.sin(table)
    sin_cache[:, ::2] = -sin_cache[:, ::2]

    cos_cache = np.cos(table)
    return sin_cache, cos_cache

def draw2():
    L = 512
    d = 128
    sin_table,cos_table = create_sin_cos_table(L, d)
#    s2,c2 = create_sin_cos_cac
    q = np.random.rand(1, d)
    #q = np.ones((1,d))
    pos_m = 256
    q_r = rotary(q, pos_m, sin_table, cos_table)
    #k = np.ones((1,d))
    k = np.random.rand(1, d)
    k_pos = np.arange(0, L)
    A = []
    for pos_n in k_pos:
        k_r = rotary(k, pos_n, sin_table, cos_table)
        alpha = np.dot(q_r, k_r.T)/np.sqrt(d)
        A.append(alpha[0])
    
    plt.plot(A)
    plt.xlabel("pos")
    plt.ylabel("$\\alpha$")
    plt.grid()
    plt.show()

if __name__=="__main__":
    draw1()
    draw2()
