import numpy as np

class Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = None
        self.init()

    def init(self):
        self.data = np.zeros(self.shape)

    def zeros(self):
        self.data = np.zeros(self.shape)

    def random(self):
        self.data = np.random.randn(*self.shape) * 0.01