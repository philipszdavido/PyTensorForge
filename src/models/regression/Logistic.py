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

    def grads(self):
        return [self.dW, self.db]

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        self.X = X
        return sigmoid(np.dot(X, self.W.data) + self.b.data)

    def backward(self, grad_output):
        self.dW.data += grad_output * self.X.data
        self.db.data += grad_output
        return grad_output * self.W.data

    def update(self, learning_rate):
        self.W.data -= learning_rate * self.dW.data
        self.b.data -= learning_rate * self.db.data