import numpy as np

class GlorotNormal:

    def __call__(self, shape):

        fan_in, fan_out = shape

        std = np.sqrt(2 / (fan_in + fan_out))

        return np.random.normal(
            0,
            std,
            shape,
        ).astype(np.float32)