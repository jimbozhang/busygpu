import threading

from numba import cuda, vectorize
from numpy.random import randint


@vectorize(['int64(int64)'], target='cuda')
def mul_self_forever(x):
    y = x
    while y is not None:
        y = x * x
    return y


def worker(device_id=0, n=45_000):
    mat = randint(1, size=(n, n))
    cuda.select_device(device_id)
    mul_self_forever(cuda.to_device(mat))


for i in range(len(cuda.gpus)):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
