import math

def L(N):
    Nc = 8.8e13
    result = math.pow(Nc/N, 0.076)
    return result

if __name__=="__main__":
    N_gpt2_medium = 354_823_167
    N_gpt2_large = 774_030_080
    L1 = L(N_gpt2_medium)
    L2 = L(N_gpt2_large)
    print(f"gpt2-medium: N={N_gpt2_medium}, L={L1}")
    print(f"gpt2-large: N={N_gpt2_large}, L={L2}")
