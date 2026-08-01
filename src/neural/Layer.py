from src.neural.Parameter import Parameter
from src.initializers import (
    GlorotUniform,
    HeUniform,
    LecunUniform,
    Zeros,
)

initializer_fns = {
    "glorot_uniform": GlorotUniform(),
    "he_uniform": HeUniform(),
    "lecun_uniform": LecunUniform(),
    "zeros": Zeros(),
}


class Layer:
    def __init__(self):
        self.built = False

        self.weights = []
        self.trainable_weights = []
        self.non_trainable_weights = []

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        return self.call(inputs)

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

    def add_weight(
        self,
        shape,
        initializer="glorot_uniform",
        trainable=True,
        name=None,
    ):
        value = initializer_fns[initializer](shape)

        parameter = Parameter(
            data=value,
            trainable=trainable,
            name=name,
        )

        self.weights.append(parameter)

        if trainable:
            self.trainable_weights.append(parameter)
        else:
            self.non_trainable_weights.append(parameter)

        return parameter

    def parameters(self):
        return self.trainable_weights