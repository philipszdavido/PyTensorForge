import numpy as np

from src.core.Tensor import Tensor


class Log:

    @staticmethod
    def forward(x):

        out = Tensor(
            np.log(x.data),
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Log",
        )

        def _backward():

            if x.requires_grad:
                x.grad += out.grad / x.data

        out._backward = _backward

        return out