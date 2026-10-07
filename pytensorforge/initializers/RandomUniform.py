import numpy as np

class RandomUniform:

    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def __call__(self, shape):
        return np.random.uniform(
            self.low,
            self.high,
            shape,
        ).astype(np.float32)