import numpy as np

from src.core.Tensor import Tensor


class Exp:

    @staticmethod
    def forward(x):

        e = np.exp(x.data)

        out = Tensor(
            e,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Exp",
        )

        def _backward():

            if x.requires_grad:
                x.grad += out.grad * e

        out._backward = _backward

        return out