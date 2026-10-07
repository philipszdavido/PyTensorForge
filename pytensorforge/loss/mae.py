from pytensorforge.loss.Loss import Loss
import numpy as np

class MAELoss(Loss):

    def forward(self, pred, target):
        return np.mean(np.abs(pred - target))

    def backward(self, pred, target):
        return np.sign(pred - target)

class MAE:

    def __call__(self, prediction, target):

        return (prediction - target).abs().mean()