import numpy as np

from pytensorforge.initializers.Initializer import Initializer


class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape, dtype=np.float32)