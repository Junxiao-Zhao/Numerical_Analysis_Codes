import numpy as np

epsilon = np.finfo(np.float64).eps

m3 = lambda e: np.matrix([[1, 1], [1, 1 + e]])

while np.linalg.matrix_rank(m3(epsilon)) < 2:
    epsilon *= 2

l = epsilon / 2
r = epsilon

for i in range(10000):
    mid = (l + r) / 2

    if np.linalg.matrix_rank(m3(mid)) == 1:
        l = mid
    else:
        r = mid

print(r)
