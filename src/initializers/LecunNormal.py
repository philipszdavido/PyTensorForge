import numpy as np


class LecunNormal:

    def __call__(self, shape):

        fan_in = shape[0]

        std = np.sqrt(1 / fan_in)

        return np.random.normal(
            0,
            std,
            shape,
        ).astype(np.float32)