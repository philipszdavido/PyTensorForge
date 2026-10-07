from pytensorforge.core.Tensor import Tensor
from pytensorforge.neural.Layer import Layer


class LSTM(Layer):

    def __init__(
        self,
        hidden_size,
        return_sequences=False,
    ):
        super().__init__()

        self.bo = None
        self.Uo = None
        self.Wo = None
        self.bg = None
        self.Ug = None
        self.Wg = None
        self.bf = None
        self.Uf = None
        self.Wf = None
        self.bi = None
        self.Ui = None
        self.Wi = None
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

    def build(self, input_shape):

        input_size = input_shape[-1]

        self.Wi = self.add_weight(
            shape=(input_size, self.hidden_size),
            initializer="glorot_uniform",
            name="Wi",
        )

        self.Ui = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            name="Ui",
        )

        self.bi = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            name="bi",
        )

        self.Wf = self.add_weight(
            shape=(input_size, self.hidden_size),
            initializer="glorot_uniform",
            name="Wf",
        )

        self.Uf = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            name="Uf",
        )

        self.bf = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            name="bf",
        )

        self.Wg = self.add_weight(
            shape=(input_size, self.hidden_size),
            initializer="glorot_uniform",
            name="Wg",
        )

        self.Ug = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            name="Ug",
        )

        self.bg = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            name="bg",
        )

        self.Wo = self.add_weight(
            shape=(input_size, self.hidden_size),
            initializer="glorot_uniform",
            name="Wo",
        )

        self.Uo = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            name="Uo",
        )

        self.bo = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            name="bo",
        )

    def call(self, x):

        batch = x.shape[0]
        timesteps = x.shape[1]

        h = Tensor.zeros(
            (batch, self.hidden_size),
            requires_grad=False,
        )

        c = Tensor.zeros(
            (batch, self.hidden_size),
            requires_grad=False,
        )

        outputs = []

        for t in range(timesteps):

            xt = x[:, t, :]

            i = (
                xt @ self.Wi +
                h @ self.Ui +
                self.bi
            ).sigmoid()

            f = (
                xt @ self.Wf +
                h @ self.Uf +
                self.bf
            ).sigmoid()

            g = (
                xt @ self.Wg +
                h @ self.Ug +
                self.bg
            ).tanh()

            o = (
                xt @ self.Wo +
                h @ self.Uo +
                self.bo
            ).sigmoid()

            c = f * c + i * g

            h = o * c.tanh()

            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return Tensor.stack(outputs, axis=1)

        return h

    def get_config(self):
        return {
            "class_name": "LSTM",
            "hidden_size": self.hidden_size,
            "return_sequences": self.return_sequences,
        }