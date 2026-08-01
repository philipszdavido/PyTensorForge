import numpy as np


class Parameter:
    def __init__(self, data, trainable=True, name=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)

        self.trainable = trainable
        self.name = name

    def zero_grad(self):
        self.grad.fill(0)

    @property
    def shape(self):
        return self.data.shape