import numpy as np

from src.core.Tensor import Tensor
from src.neural.Layer import Layer


class Dropout(Layer):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def build(self, input_shape):
        pass

    def call(self, x):

        if not self.training or self.p <= 0:
            return x

        keep = 1 - self.p

        mask = np.random.binomial(
            1,
            keep,
            size=x.shape,
        ).astype(np.float32)

        mask = Tensor(
            mask / keep,
            requires_grad=False,
        )

        return x * mask