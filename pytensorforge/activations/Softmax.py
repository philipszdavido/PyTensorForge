from pytensorforge.backend import xp
from pytensorforge.core.Tensor import Tensor


# class Softmax:
#
#     @staticmethod
#     def forward(x):
#
#         shifted = x.data - xp.max(x.data, axis=-1, keepdims=True)
#
#         exp = xp.exp(shifted)
#
#         probs = exp / xp.sum(exp, axis=-1, keepdims=True)
#
#         out = Tensor(
#             probs,
#             requires_grad=x.requires_grad,
#             parents=(x,),
#             op="Softmax",
#         )
#
#         def _backward():
#
#             if not x.requires_grad:
#                 return
#
#             grad = xp.empty_like(probs)
#
#             for i in range(len(probs)):
#                 p = probs[i].reshape(-1, 1)
#
#                 jacobian = xp.diagflat(p) - p @ p.T
#
#                 grad[i] = jacobian @ out.grad[i]
#
#             x.grad += grad
#
#         out._backward = _backward
#
#         return out
#
#     def __call__(self, x):
#         return self.forward(x)

from pytensorforge.activations.Activation import Activation


class Softmax(Activation):

    @staticmethod
    def forward(x, axis=-1):

        shifted = x.data - xp.max(x.data, axis=axis, keepdims=True)

        exp = xp.exp(shifted)

        probs = exp / xp.sum(exp, axis=axis, keepdims=True)

        out = Tensor(
            probs,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Softmax",
        )

        def _backward():

            if not x.requires_grad:
                return

            grad = out.grad

            dot = xp.sum(grad * probs, axis=axis, keepdims=True)

            x.grad += probs * (grad - dot)

        out._backward = _backward

        return out

    def __call__(self, x):
        return Softmax.forward(x)