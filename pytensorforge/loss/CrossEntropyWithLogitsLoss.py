from pytensorforge.backend import xp
from pytensorforge.core.Tensor import Tensor


class CrossEntropyWithLogitsLoss:

    def __call__(self, logits, target):

        batch = logits.shape[0]

        shifted = logits.data - xp.max(
            logits.data,
            axis=-1,
            keepdims=True,
        )

        logsumexp = xp.log(
            xp.sum(xp.exp(shifted), axis=-1)
        )

        indices = target.data.astype(xp.int64)

        loss = (
            -shifted[xp.arange(batch), indices]
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

            exp = xp.exp(shifted)

            probs = exp / xp.sum(
                exp,
                axis=-1,
                keepdims=True,
            )

            grad = probs

            grad[xp.arange(batch), indices] -= 1

            grad /= batch

            logits.grad += grad * out.grad

        out._backward = _backward

        return out