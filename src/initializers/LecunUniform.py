import numpy as np

from src.initializers.Initializer import Initializer


class LecunUniform(Initializer):
    def __call__(self, shape):
        fan_in = shape[0]
        limit = np.sqrt(3.0 / fan_in)
        return np.random.uniform(
            -limit,
            limit,
            shape,
        ).astype(np.float32)