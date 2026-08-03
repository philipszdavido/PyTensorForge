import numpy as np

class RandomNormal:

    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        return np.random.normal(
            self.mean,
            self.stddev,
            shape,
        ).astype(np.float32)