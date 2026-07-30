import numpy as np

from src.core.Tensor import Tensor
from src.math.sigmoid import sigmoid


class Logistic:
    def __init__(self, weight_shape, bias_shape):
        self.X = None
        self.W = Tensor(weight_shape)
        self.W.random()

        self.b = Tensor(bias_shape)
        self.b.random()

        self.dW = Tensor(weight_shape)
        self.db = Tensor(bias_shape)

    def forward(self, X):
        self.X = X
        return sigmoid(np.dot(X, self.W.data) + self.b.data)

    def backward(self, gradient):
        self.dW.data = gradient * self.X.data
        self.db.data = gradient

    def update(self, learning_rate):
        self.W.data -= learning_rate * self.dW.data
        self.b.data -= learning_rate * self.db.data