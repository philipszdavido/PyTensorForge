from src.activations import activation_fns
from src.models.gpt.attention import CausalSelfAttention
from src.models.transformers.Dropout import Dropout
from src.models.transformers.LayerNorm import LayerNorm
from src.neural.Dense import Dense
from src.neural.Layer import Layer


class GPTBlock(Layer):

    def __init__(self, config, rope=None):
        super().__init__()

        self.config = config

        self.norm1 = LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, rope=rope)

        self.norm2 = LayerNorm(config.d_model, eps=config.norm_eps)
        self.fc1 = Dense(config.ff_dim)
        self.fc2 = Dense(config.d_model)

        self.dropout = Dropout(config.dropout)
        self.activation = activation_fns[config.activation]

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape)

        self.norm2.build(input_shape)
        self.fc1.build(input_shape)

        ff_shape = (*input_shape[:-1], self.fc1.units)
        self.fc2.build(ff_shape)

        self.dropout.build(input_shape)

        self.built = True

    def call(self, x, mask=None):
        h = self.norm1.call(x)
        h = self.attn.call(h, mask=mask)
        x = x + self.dropout.call(h)

        h = self.norm2.call(x)
        h = self.fc2.call(self.activation(self.fc1.call(h)))
        x = x + self.dropout.call(h)

        return x

    def parameters(self):
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def state_dict(self):
        return {
            "norm1": self.norm1.state_dict(),
            "attn": self.attn.state_dict(),
            "norm2": self.norm2.state_dict(),
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
        }

    def load_state_dict(self, state):
        self.norm1.load_state_dict(state["norm1"])
        self.attn.load_state_dict(state["attn"])
        self.norm2.load_state_dict(state["norm2"])
        self.fc1.load_state_dict(state["fc1"])
        self.fc2.load_state_dict(state["fc2"])
