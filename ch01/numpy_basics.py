import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x + y)
print(x - y)
print(x * y)
print(x / y)

A = np.array([[1, 2], [3, 4]])
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A.shape)
print(A.dtype)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X[0])
print(X[0][1])
print(X > 15)
print(X[X > 15])
