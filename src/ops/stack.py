import numpy as np

from src.core.Tensor import Tensor


class Stack:

    @staticmethod
    def forward(tensors, axis=0):

        data = np.stack(
            [t.data for t in tensors],
            axis=axis,
        )

        out = Tensor(
            data,
            requires_grad=any(t.requires_grad for t in tensors),
            parents=tuple(tensors),
            op="Stack",
        )

        def _backward():

            for i, tensor in enumerate(tensors):

                if not tensor.requires_grad:
                    continue

                grad = np.take(
                    out.grad,
                    indices=i,
                    axis=axis,
                )

                tensor.grad += grad

        out._backward = _backward

        return out