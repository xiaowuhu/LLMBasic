import numpy as np

def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return a

def MinMaxScalar(k, q):
    k_min = np.min(k, axis=0)
    k_max = np.max(k, axis=0)
    k = (k - k_min) / (k_max - k_min)
    q = (q - k_min) / (k_max - k_min)
    return k, q

if __name__=="__main__":
    k = np.array([[55,13],[62,8],[78,16],[97,12]])
    k = np.array([[55,7],[62,12],[78,4],[97,8]]) # k2 = 20 - k2
    v = np.array([[320],[355],[428],[478]])
    q = np.array([[66,10]])   # A=403
    #q = np.array([[90,14]])  # A=418, 406
    #q = np.array([[90,10]])  # A=416, 410
    print("q =", q)
    print("k =", k)
    k, q = MinMaxScalar(k, q)
    print("q =", q)
    print("k =", k)

    z = np.dot(q, k.T)
    print("q · k.T=", z)
    #a = softmax(z/np.sqrt(2))
    s = softmax(z)
    print("softmax=", s)
    a = np.dot(s, v)
    print("A=", a)
