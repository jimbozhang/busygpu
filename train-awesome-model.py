import numpy as np
from numba import vectorize


@vectorize(['int64(int64)'], target='cuda')
def mulself(a):
    k = 1000
    for i in range(k):
        a = a + a * a
    return a


n = 30_000
mat = np.arange(n * n).reshape(n, n)
while True:
    mulself(mat)
