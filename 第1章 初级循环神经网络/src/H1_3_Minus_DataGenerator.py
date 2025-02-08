
import numpy as np
import os

# n=16,32,64
def create_data(n):
    count = 1
    a = []
    b = []
    c = []
    for i in range(0,n):
        for j in range(0,i+1):
            #print(count,i,j)
            count+=1
            k = i - j
            a.append(i)
            b.append(j)
            c.append(k)
    return a,b,c

def generate_data_bin4():
    a,b,c = create_data(16)
    binary8 = np.unpackbits(np.array([range(16)],dtype=np.uint8).T,axis=1)
    # 只取后4位，最大值15
    bin4 = binary8[:,4:8]
    # 倒排二进制顺序，如1=0001, -> 1000
    for i in range(bin4.shape[0]):
        bin4[i] = bin4[i][::-1]
        print(i, bin4[i])
    
    count = len(a)
    X = np.zeros((count,4,2),dtype=np.uint8)
    Y = np.zeros((count,4),dtype=np.uint8)
    for i in range(count):
        bin_a = bin4[a[i]]
        bin_b = bin4[b[i]]
        bin_c = bin4[c[i]]
        for j in range(4):
            X[i,j] = [bin_a[j],bin_b[j]]
            Y[i,j] = bin_c[j]

    print(X.shape)
    print(Y.shape)
    return X, Y

def generate_data_oct():
    x1, x2, y = create_data(16)
    X1 = np.array(x1).reshape(-1,1)
    X2 = np.array(x2).reshape(-1,1)
    Y = np.array(y).reshape(-1,1)
    return np.hstack((X1, X2, Y))

def generate_data_bin(n):
    a,b,c = create_data(2**n)
    binary8 = np.unpackbits(np.array([range(2**n)],dtype=np.uint8).T,axis=1)
    # 只取后4位，最大值15
    bin5 = binary8[:,8-n:8]
    # 倒排二进制顺序，如1=0001, -> 1000
    for i in range(bin5.shape[0]):
        bin5[i] = bin5[i][::-1]
    
    count = len(a)
    X = np.zeros((count,n,2),dtype=np.uint8)
    Y = np.zeros((count,n),dtype=np.uint8)
    for i in range(count):
        bin_a = bin5[a[i]]
        bin_b = bin5[b[i]]
        bin_c = bin5[c[i]]
        for j in range(n):
            X[i,j] = [bin_a[j],bin_b[j]]
            Y[i,j] = bin_c[j]

    print(X.shape)
    print(Y.shape)
    return X, Y

def save_txt(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%i")

def save_npz(filename, X, Y):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", filename)
    np.savez(filename, data=X, label=Y)    

if __name__=='__main__':
    train = generate_data_oct()   
    save_txt(train, "minus_train_oct.txt")
    save_txt(train, "minus_test_oct.txt")
    
    train_x, train_y = generate_data_bin(4)
    save_npz("minus_train_bin.npz", train_x, train_y)

    train_x, train_y = generate_data_bin(5)
    save_npz("minus_test_bin_5.npz", train_x, train_y)
