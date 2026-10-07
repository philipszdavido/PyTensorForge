import numpy as np

from pytensorforge.core.Tensor import Tensor


class CrossEntropyWithLogitsLoss:

    def __call__(self, logits, target):

        batch = logits.shape[0]

        shifted = logits.data - np.max(
            logits.data,
            axis=-1,
            keepdims=True,
        )

        logsumexp = np.log(
            np.sum(np.exp(shifted), axis=-1)
        )

        indices = target.data.astype(np.int64)

        loss = (
            -shifted[np.arange(batch), indices]
            + logsumexp
        ).mean()

        out = Tensor(
            loss,
            requires_grad=logits.requires_grad,
            parents=(logits,),
            op="CrossEntropyWithLogits",
        )

        def _backward():

            if not logits.requires_grad:
                return

            exp = np.exp(shifted)

            probs = exp / np.sum(
                exp,
                axis=-1,
                keepdims=True,
            )

            grad = probs

            grad[np.arange(batch), indices] -= 1

            grad /= batch

            logits.grad += grad * out.grad

        out._backward = _backward

        return out