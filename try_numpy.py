import numpy as np

a = np.array([
    [1, 2, 3],
    [3, 1, 5],
    [4, 5, 1]
])

min = np.min(a, axis=0)
print(min)
