import numpy as np

from src.core.Tensor import Tensor


# class Parameter:
#     def __init__(self, data, trainable=True, name=None):
#         self.data = np.asarray(data, dtype=np.float32)
#         self.grad = np.zeros_like(self.data)
#
#         self.trainable = trainable
#         self.name = name
#
#     def zero_grad(self):
#         self.grad.fill(0)
#
#     @property
#     def shape(self):
#         return self.data.shape

class Parameter(Tensor):

    def __init__(self, data, trainable=True, name=None):
        super().__init__(
            data,
            requires_grad=True,
        )

        self.name = name
        self.trainable = trainable