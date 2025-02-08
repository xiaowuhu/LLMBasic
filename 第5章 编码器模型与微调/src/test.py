import random

a = []
b = []
c = []
d = []
for i in range(1000):
    if random.random() < 0.15:
        a.append(i)
        if random.random() < 0.8:
            b.append(i)
        elif random.random() < 0.5:
            c.append(i)
        else:
            d.append(i)
        
print(len(a)/1000)
print(len(b)/1000)
print(len(c)/1000)
print(len(d)/1000)
