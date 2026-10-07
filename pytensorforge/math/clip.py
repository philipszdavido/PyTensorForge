import numpy as np

from pytensorforge.core.Tensor import Tensor


class Clip:

    @staticmethod
    def forward(x, min_value, max_value):

        clipped = np.clip(x.data, min_value, max_value)

        out = Tensor(
            clipped,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Clip",
        )

        def _backward():

            if not x.requires_grad:
                return

            mask = (
                (x.data >= min_value)
                & (x.data <= max_value)
            ).astype(np.float32)

            x.grad += out.grad * mask

        out._backward = _backward

        return out

    def __call__(self, x, min_value, max_value):
        return self.forward(x, min_value, max_value)