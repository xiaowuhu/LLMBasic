
import torch

def normal_softmax(X):
    X_exp_sum = X.exp().sum()
    softmax = torch.exp(X) / X_exp_sum
    print("普通softmax:")
    print(softmax)

def safe_softmax(X):
    X_max = X.max()
    X_exp_sum_sub_max = torch.exp(X-X_max).sum()
    softmax = torch.exp(X - X_max) / X_exp_sum_sub_max
    print("安全softmax:")
    print("最大值:", X_max)
    print(softmax)

def online_softmax(X):
    X_pre = X[:-1]
    print("在线计算softmax:")
    print('input x')
    print(X)
    print(X_pre)
    print(X[-1])

    # we calculative t-1 time Online Softmax
    X_max_pre = X_pre.max()
    X_sum_pre = torch.exp(X_pre - X_max_pre).sum()

    # we calculative t time Online Softmax
    X_max_cur = torch.max(X_max_pre, X[-1]) # X[-1] is new data
    X_sum_cur = X_sum_pre * torch.exp(X_max_pre - X_max_cur) + torch.exp(X[-1] - X_max_cur)

    # final we calculative online softmax
    X_online_softmax = torch.exp(X - X_max_cur) / X_sum_cur
    print(X_online_softmax)

def block_online_softmax(X):
    print("在线块计算softmax:")

    X_block = torch.split(X, split_size_or_sections = 3 , dim = 0) 
    print(X)
    print(X_block)

    # we parallel calculate  different block max & sum
    X_block_0_max = X_block[0].max()
    X_block_0_sum = torch.exp(X_block[0] - X_block_0_max).sum()

    X_block_1_max = X_block[1].max()
    X_block_1_sum = torch.exp(X_block[1] - X_block_1_max).sum()

    # online block update max & sum
    X_block_1_max_update = torch.max(X_block_0_max, X_block_1_max) # X[-1] is new data
    X_block_1_sum_update = X_block_0_sum * torch.exp(X_block_0_max - X_block_1_max_update) \
                        + torch.exp(X_block[1] - X_block_1_max_update).sum() # block sum

    X_block_online_softmax = torch.exp(X - X_block_1_max_update) / X_block_1_sum_update
    print(X_block_online_softmax)


def batch_online_softmax():
    print("批量计算方法:")

    NEG_INF = -1e10  # -infinity
    EPSILON = 1e-10

    Q_LEN = 6
    K_LEN = 6
    Q_BLOCK_SIZE = 3
    KV_BLOCK_SIZE = 3
    P_DROP = 0.2

    Tr = Q_LEN // Q_BLOCK_SIZE
    Tc = K_LEN // KV_BLOCK_SIZE

    Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
    K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[..., None]
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

    # step 4
    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

    # step 5
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    # step 6
    for j in range(Tc):
        # step 7
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        # step 8
        for i in range(Tr):
            # step 9
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            # step 10
            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)

            # step 11
            mask = S_ij.ge(0.5)
            S_ij = torch.masked_fill(S_ij, mask, value=0)
            
            # step 12
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

            # step 13
            mi_new = torch.maximum(m_block_ij, mi)

            li_new = torch.exp(mi - mi_new) * li + \
                    torch.exp(m_block_ij - mi_new) * l_block_ij

            # step 14
            m = torch.nn.Dropout(p=P_DROP)
            P_ij_Vj = m(P_ij_Vj)

            # Step 15
            O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                        + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            print(f'-----------Attention : Q{i}xK{j}---------')
            print(O_BLOCKS[i].shape)
            print(O_BLOCKS[0])
            print(O_BLOCKS[1])
            print('\n')

            # step 16
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new

    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)    

if __name__=="__main__":
    X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
    normal_softmax(X)
    safe_softmax(X)
    online_softmax(X)
    block_online_softmax(X)
    batch_online_softmax()
