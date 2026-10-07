import numpy as np

class Ones:

    def __call__(self, shape):
        return np.ones(shape, dtype=np.float32)