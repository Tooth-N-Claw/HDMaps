import numpy as np

a = np.array([[1, 2], [3, 4]])

a[0:2, 0:2] *= 2
print(a)