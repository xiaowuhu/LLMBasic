import numpy as np

x = np.array([[1,2,3,4],[5,6,7,8]])
y = np.array([[9,13],[10,14],[11,15],[12,16]])
z = np.dot(x,y)
print(x)
print(y)
print(z)


y1 = np.array([[9],[10],[11],[12]])
y2 = np.array([[13],[14],[15],[16]])
z1 = np.dot(x,y1)
z2 = np.dot(x,y2)
print(np.concatenate((z1,z2), axis=1))

x1 = np.array([[1,2],[5,6]])
x2 = np.array([[3,4],[7,8]])
y1 = np.array([[9,13],[10,14]])
y2 = np.array([[11,15],[12,16]])
z1 = np.dot(x1,y1)
z2 = np.dot(x2,y2)
z = z1+z2
print(z1)
print(z2)
print(z)
