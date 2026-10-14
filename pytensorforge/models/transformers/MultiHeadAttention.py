from pytensorforge.backend import assign, copy_to_numpy, xp
from pytensorforge.core.Tensor import Tensor
from pytensorforge.neural.Layer import Layer


class MultiHeadAttention(Layer):

    def __init__(
        self,
        d_model,
        num_heads,
        bias=True,
    ):
        super().__init__()

        self.Wv = None
        self.Wk = None
        self.Wq = None
        self.Wo = None
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.bias = bias

    def build(self, input_shape):

        self.Wq = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            name="Wq",
        )

        self.Wk = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            name="Wk",
        )

        self.Wv = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            name="Wv",
        )

        self.Wo = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            name="Wo",
        )

    def call(self, x, mask=None):

        batch = x.shape[0]
        seq = x.shape[1]

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        Q = Q.reshape(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )

        K = K.reshape(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )

        V = V.reshape(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2)

        # scores = scores / math.sqrt(self.head_dim)
        scores = scores / Tensor(self.head_dim).sqrt()
        # mask = xp.triu(
        #     xp.ones((seq, seq)),
        #     k=1
        # ).astype(bool)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = scores.softmax(axis=-1)

        context = weights @ V

        context = context.transpose(0, 2, 1, 3)

        context = context.reshape(
            batch,
            seq,
            self.d_model,
        )

        output = context @ self.Wo

        return output

    def state_dict(self):
        return {
            "Wq": copy_to_numpy(self.Wq.data),
            "Wk": copy_to_numpy(self.Wk.data),
            "Wv": copy_to_numpy(self.Wv.data),
            "Wo": copy_to_numpy(self.Wo.data),
        }

    def load_state_dict(self, state):
        assign(self.Wq.data, state["Wq"])
        assign(self.Wk.data, state["Wk"])
        assign(self.Wv.data, state["Wv"])
        assign(self.Wo.data, state["Wo"])