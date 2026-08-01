import numpy as np

from src.activations.Activation import Activation

class SELU(Activation):
    SCALE = 1.0507009873554805
    ALPHA = 1.6732632423543772

    def __call__(self, x):
        return self.SCALE * np.where(
            x > 0,
            x,
            self.ALPHA * (np.exp(x) - 1),
        )