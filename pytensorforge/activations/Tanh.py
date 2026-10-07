import numpy as np

from pytensorforge.activations.Activation import Activation
from pytensorforge.core.Tensor import Tensor


class Tanh(Activation):

    @staticmethod
    def forward(x):
        t = np.tanh(x.data)

        out = Tensor(
            t,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Tanh",
        )

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * (1 - t * t)

        out._backward = _backward

        return out

    def __call__(self, x):
        return x.tanh()