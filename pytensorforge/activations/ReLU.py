import numpy as np

from pytensorforge.activations.Activation import Activation
from pytensorforge.core.Tensor import Tensor


class ReLU(Activation):

    @staticmethod
    def forward(x):
        out = Tensor(
            np.maximum(0, x.data),
            requires_grad=x.requires_grad,
            parents=(x,),
            op="ReLU",
        )

        def backward():
            if x.requires_grad:
                x.grad += out.grad * (x.data > 0)

        out._backward = backward

        return out

    def __call__(self, x):
        return x.relu()