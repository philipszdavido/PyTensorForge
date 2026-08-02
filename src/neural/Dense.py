import numpy as np

from src.neural.Layer import Layer
from src.initializers import (
    GlorotUniform,
    HeUniform,
    LecunUniform,
    Zeros,
)
from src.activations import (
    ReLU,
    Tanh,
    Sigmoid,
    ELU,
    SELU,
)

initializer_fns = {
    "glorot_uniform": GlorotUniform(),
    "he_uniform": HeUniform(),
    "lecun_uniform": LecunUniform(),
    "zeros": Zeros(),
}

activation_fns = {
    "relu": ReLU(),
    "tanh": Tanh(),
    "sigmoid": Sigmoid(),
    "elu": ELU(),
    "selu": SELU(),
}


class Dense(Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.kernel = self.add_weight(
            # row = in_features, column = units
            shape=(in_features, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )

        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call_(self, inputs):
        output = np.matmul(inputs, self.kernel.data) + self.bias.data

        if self.activation is not None:
            output = activation_fns[self.activation](output)

        return output

    def call(self, inputs):
        output = inputs @ self.kernel
        output = output + self.bias

        if self.activation is not None:
            output = activation_fns[self.activation](output)

        return output