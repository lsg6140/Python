import numpy as np

a = np.array([0,1,2])
print(a.shape)
print(a)
print(a.T)

print(np.dot(a,a))
print(np.dot(a,a.T))

print(a @ a)
