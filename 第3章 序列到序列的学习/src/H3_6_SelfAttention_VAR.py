import numpy as np

def test_var():
    x = np.random.normal(loc=2, scale=2, size=(1, 100))
    y = np.random.normal(loc=2, scale=2, size=(1, 100))
    z = x * y
    z2 = x * y / 10
    print("形状: x=", x.shape, "y=", y.shape, "z=", z.shape, "z2=", z2.shape)
    print("方差: x=%.2f"%x.var(), "y=%.2f"%y.var(), "z=%.2f"%z.var(), "z2=%.2f"%z2.var())
    print("均值: x=%.2f"%x.mean(), "y=%.2f"% y.mean(), "z=%.2f"%z.mean(), "z2=%.2f"%z2.mean())
    

if __name__=="__main__":
    test_var()
