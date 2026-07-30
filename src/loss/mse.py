from src.loss.Loss import Loss
import numpy as np

class MSELoss(Loss):

    def forward(self, pred, target):
        return 0.5 * np.mean((pred - target) ** 2)

    def backward(self, pred, target):
        return pred - target
