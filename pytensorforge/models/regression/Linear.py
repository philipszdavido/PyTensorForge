import numpy as np

from pytensorforge.core.Tensor import Tensor


# y = XW + B
class Linear:
    def __init__(self, weight_shape, bias_shape):

        self.W = Tensor.random(weight_shape)

        self.b = Tensor.random(bias_shape)

        self.dW = Tensor.zeros(weight_shape)
        self.db = Tensor.zeros(bias_shape)

        self.X = None

    def normalize(self, X):
        norm = np.linalg.norm(X)
        return X if norm == 0 else X / norm

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.data) + self.b.data

    def grads(self):
        return [self.dW, self.db]

    def parameters(self):
        return [self.W, self.b]

    def backward(self, grad_output):
        # L = pred - target
        # u = pred - target
        # L = u
        # dL/du = 1
        # dL/dpred = dL/du * du/dpred
        # du/dpred = 1 - 0 = 1
        # dL/dpred = u = pred - target
        # how do W affect the L?
        # dL/dW = dL/dpred * dpred/dW
        # dpred/dW = X
        # dL/dW = (pred - target) * X
        self.dW.data += grad_output * self.X.data
        self.db.data += grad_output
        return grad_output * self.W.data

    def update(self, learning_rate):
        self.W.data -= learning_rate * self.dW.data
        self.b.data -= learning_rate * self.db.data