import numpy as np

from pytensorforge.neural.Layer import Layer
from pytensorforge.core.Tensor import Tensor


class Embedding(Layer):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.weight = None

    def build(self, input_shape):

        self.weight = self.add_weight(
            shape=(
                self.vocab_size,
                self.embedding_dim,
            ),
            initializer="random_normal",
            name="embedding",
        )

    def call(self, x):

        indices = x.data.astype(np.int64)

        out = Tensor(
            self.weight.data[indices],
            requires_grad=True,
            parents=(self.weight,),
            op="Embedding",
        )

        def _backward():

            if not self.weight.requires_grad:
                return

            np.add.at(
                self.weight.grad,
                indices,
                out.grad,
            )

        out._backward = _backward

        return out

    def compute_output_shape(self, input_shape):
        batch, seq = input_shape

        return (
            batch,
            seq,
            self.embedding_dim,
        )