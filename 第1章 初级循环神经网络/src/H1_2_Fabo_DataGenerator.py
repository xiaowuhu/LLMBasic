import os
import numpy as np

def create_data(count):
    f_n1 = 0
    f_n2 = 1
    X = np.zeros((count, 2))
    Y = np.zeros((count, 1))
    for i in range(count): 
        f_n3 = f_n1 + f_n2
        X[i,0] = f_n1
        X[i,1] = f_n2
        Y[i,0] = f_n3
        f_n1 = f_n2
        f_n2 = f_n3
    
    return np.hstack((X, Y))

def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.2f")

if __name__=='__main__':
    np.random.seed(17)
    xy = create_data(23)
    save_data(xy, "fabo_train.txt")
    xy = create_data(30)
    save_data(xy, "fabo_test.txt")
    print("done")
