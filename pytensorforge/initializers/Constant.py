import numpy as np

class Constant:

    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value, dtype=np.float32)