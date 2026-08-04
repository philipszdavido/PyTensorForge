from src.neural.Layer import Layer
from src.activations import activation_fns


class Dense(Layer):
    def __init__(self, units, activation=None,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "zeros",
        trainable = True,
    ):
        super().__init__()
        self.units = units
        self.activation = activation

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.kernel = self.add_weight(
            # row = in_features, column = units
            shape=(in_features, self.units),
            initializer=self.kernel_initializer,
            trainable=self.trainable,
            name="kernel",
        )

        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            trainable=self.trainable,
            name="bias",
        )

    def call(self, inputs):
        output = inputs @ self.kernel
        output = output + self.bias

        if self.activation is not None:
            output = activation_fns[self.activation](output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        return {
            "units": self.units,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        }