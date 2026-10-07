from pytensorforge.activations import activation_fns
from pytensorforge.models.transformers.Dropout import Dropout
from pytensorforge.models.transformers.LayerNorm import LayerNorm
from pytensorforge.models.transformers.MultiHeadAttention import MultiHeadAttention
from pytensorforge.neural.Dense import Dense
from pytensorforge.neural.Layer import Layer


class TransformerBlock(Layer):

    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.fc1 = Dense(ff_dim)
        self.fc2 = Dense(d_model)

        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(0.1)

    def call(self, x):
        h = self.attn(x)
        x = self.norm1(x + self.dropout(h))
        # x = self.norm1(x + h)

        h = self.fc2(activation_fns["gelu"](self.fc1(x)))
        x = self.norm2(x + self.dropout(h))
        # x = self.norm2(x + h)

        return x

    def build(self, input_shape):
        self.attn.build(input_shape)

        self.norm1.build(input_shape)

        self.fc1.build(input_shape)

        ff_shape = (*input_shape[:-1], self.fc1.units)

        self.fc2.build(ff_shape)

        self.norm2.build(input_shape)

        self.dropout.build(input_shape)

        self.built = True

    def parameters(self):
        params = []

        params.extend(self.attn.parameters())

        params.extend(self.norm1.parameters())

        params.extend(self.fc1.parameters())

        params.extend(self.fc2.parameters())

        params.extend(self.norm2.parameters())

        return params

    def state_dict(self):
        return {
            "attn": self.attn.state_dict(),
            "norm1": self.norm1.state_dict(),
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
            "norm2": self.norm2.state_dict(),
        }

    def load_state_dict(self, state):
        self.attn.load_state_dict(state["attn"])
        self.norm1.load_state_dict(state["norm1"])
        self.fc1.load_state_dict(state["fc1"])
        self.fc2.load_state_dict(state["fc2"])
        self.norm2.load_state_dict(state["norm2"])