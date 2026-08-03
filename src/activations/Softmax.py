import numpy as np

from src.core.Tensor import Tensor


class Softmax:

    @staticmethod
    def forward(x):

        shifted = x.data - np.max(x.data, axis=-1, keepdims=True)

        exp = np.exp(shifted)

        probs = exp / np.sum(exp, axis=-1, keepdims=True)

        out = Tensor(
            probs,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Softmax",
        )

        def _backward():

            if not x.requires_grad:
                return

            grad = np.empty_like(probs)

            for i in range(len(probs)):
                p = probs[i].reshape(-1, 1)

                jacobian = np.diagflat(p) - p @ p.T

                grad[i] = jacobian @ out.grad[i]

            x.grad += grad

        out._backward = _backward

        return out

    def __call__(self, x):
        return self.forward(x)