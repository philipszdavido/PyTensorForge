from pytensorforge.loss.Loss import Loss
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

class BinaryCrossEntropy:

    def __call__(self, prediction, target):

        eps = 1e-7

        prediction = prediction.clip(eps, 1 - eps)

        loss = -(
            target * prediction.log()
            + (1 - target) * (1 - prediction).log()
        )

        return loss.mean()