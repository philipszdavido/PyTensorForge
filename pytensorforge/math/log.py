from pytensorforge.backend import xp
from pytensorforge.core.Tensor import Tensor


class Log:

    @staticmethod
    def forward(x):

        out = Tensor(
            xp.log(x.data),
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Log",
        )

        def _backward():

            if x.requires_grad:
                x.grad += out.grad / x.data

        out._backward = _backward

        return out