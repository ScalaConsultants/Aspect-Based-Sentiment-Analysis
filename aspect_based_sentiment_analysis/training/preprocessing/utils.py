from typing import Iterator
import numpy as np


def random(dtype) -> Iterator:
    def infinite_loop():
        while True:
            x = np.random.random_sample()
            yield bool(int(round(x))) if dtype is bool else x
    iterable = infinite_loop()
    return iter(iterable)
