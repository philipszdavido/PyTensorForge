from pytensorforge.neural.Layer import Layer
from pytensorforge.activations import Tanh
from pytensorforge.core.Tensor import Tensor


class RNN(Layer):

    def __init__(
        self,
        hidden_size,
        return_sequences=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.Wx = None
        self.Wh = None
        self.bh = None

    def build(self, input_shape):
        input_size = input_shape[-1]

        self.Wx = self.add_weight(
            shape=(input_size, self.hidden_size),
            initializer="glorot_uniform",
            trainable=True,
            name="Wx",
        )

        self.Wh = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            trainable=True,
            name="Wh",
        )

        self.bh = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
            name="bh",
        )

    def call(self, x):

        batch_size = x.shape[0]
        time_steps = x.shape[1]

        h = Tensor.zeros(
            (batch_size, self.hidden_size),
            requires_grad=False,
        )

        outputs = []

        for t in range(time_steps):

            xt = x[:, t, :]

            h = Tanh.forward(
                xt @ self.Wx +
                h @ self.Wh +
                self.bh
            )

            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return Tensor.stack(outputs, axis=1)

        return h

    def compute_output_shape(self, input_shape):

        batch = input_shape[0]

        return (
            batch,
            self.hidden_size,
        )