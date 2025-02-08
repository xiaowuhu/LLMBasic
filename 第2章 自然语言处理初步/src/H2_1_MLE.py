import numpy as np
import matplotlib.pyplot as plt

def func(n1, n2):
    Y = []
    T = []
    for theta in np.linspace(0,1,101):
        T.append(theta)
        y = np.power(theta, n1) * np.power(1-theta, n2)
        Y.append(y)
    return T, Y

if __name__=="__main__":
    T, Y = func(7,3)
    plt.plot(T, Y)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r"$L(\theta|x)$")
    plt.grid()
    plt.show()
    print("theta=0.7, L(theta|x)=", Y[70])

    T, Y = func(67,33)
    loc = np.argmax(Y)
    print("max L=", Y[loc], "theta=", T[loc] )

