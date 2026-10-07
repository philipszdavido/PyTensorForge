import numpy as np

from pytensorforge.neural.Layer import Layer


class LayerNorm(Layer):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.beta = None
        self.gamma = None
        self.d_model = d_model
        self.eps = eps

    def call_(self, x, mask=None):
        mean = x.mean(axis=-1, keepdims=True)

        # var = ((x - mean) ** 2).mean(
        #     axis=-1,
        #     keepdims=True,
        # )
        diff = x - mean
        var = (diff * diff).mean(axis=-1, keepdims=True)

        x_hat = (x - mean) / (var + self.eps).sqrt()

        return self.gamma * x_hat + self.beta

    def call(self, x, mask=None):
        # mean = x.mean(axis=-1, keepdims=True)
        #
        # var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        #
        # y = self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta
        # return y
        mean = x.mean(axis=-1, keepdims=True)

        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        xhat = (x - mean) / (var + self.eps).sqrt()

        return self.gamma * xhat + self.beta

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.d_model,),
            initializer="ones",
            name="gamma",
        )

        self.beta = self.add_weight(
            shape=(self.d_model,),
            initializer="zeros",
            name="beta",
        )
