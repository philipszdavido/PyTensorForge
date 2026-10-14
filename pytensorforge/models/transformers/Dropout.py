from pytensorforge.backend import host_rng_sample_mask
from pytensorforge.core.Tensor import Tensor
from pytensorforge.neural.Layer import Layer


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

        mask = host_rng_sample_mask(x.shape, keep)
        mask = Tensor(
            mask / keep,
            requires_grad=False,
        )

        return x * mask