import torch

def formula(r):
    q = round(r/S + Z)
    return q

def test1():
    r0 = 0
    r1 = -0.99
    r2 = -1.01
    r3 = 1.67
    
    q0 = formula(r0)
    q1 = formula(r1)
    q2 = formula(r2)
    q3 = formula(r3)
    print(f'{r0} -> {q0}')
    print(f'{r1} -> {q1}')
    print(f'{r2} -> {q2}')
    print(f'{r3} -> {q3}')
    
if __name__=="__main__":
    r_min = -3.4e38
    r_max = 3.4e38
    q_min = -128
    q_max = 127
    S = (r_max - r_min) / (q_max - q_min)
    Z = q_min - r_min / S
    test1()

    r_min = -9.99
    r_max = 10.51
    S = (r_max - r_min) / (q_max - q_min)
    Z = q_min - r_min / S
    test1()
