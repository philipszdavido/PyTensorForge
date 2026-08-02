import numpy as np
from src.loss.Loss import Loss

class MSELoss(Loss):

    def forward(self, pred, target):
        return 0.5 * np.mean((pred - target) ** 2)

    def backward(self, pred, target):
        return pred - target
