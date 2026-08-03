from src.core.Tensor import Tensor


class CrossEntropyLoss:

    def __call__(self, logits, target):

        probs = logits.softmax()

        batch = logits.shape[0]

        indices = target.data.astype(int)

        p = probs[
            Tensor.arange(batch),
            indices,
        ]

        return -(p.log()).mean()

# class CrossEntropyLoss:
#
#     def __call__(self, logits, target):
#
#         probs = logits.softmax()
#
#         batch = logits.shape[0]
#
#         loss = 0
#
#         for i in range(batch):
#             loss += -probs[i][int(target.data[i])].log()
#
#         return loss / batch
