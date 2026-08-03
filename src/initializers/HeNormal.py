import numpy as np

class HeNormal:

    def __call__(self, shape):

        fan_in = shape[0]

        std = np.sqrt(2 / fan_in)

        return np.random.normal(
            0,
            std,
            shape,
        ).astype(np.float32)