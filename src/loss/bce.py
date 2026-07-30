from src.loss.Loss import Loss
import numpy as np

class BCELoss(Loss):

    def forward(self, pred, target):
        eps = 1e-8
        pred = np.clip(pred, eps, 1 - eps)

        return -np.mean(
            target * np.log(pred) +
            (1 - target) * np.log(1 - pred)
        )

    def backward(self, pred, target):
        eps = 1e-8
        pred = np.clip(pred, eps, 1 - eps)

        return (pred - target) / (pred * (1 - pred))