from pytensorforge.loss.Loss import Loss
from pytensorforge.backend import xp
class MAELoss(Loss):

    def forward(self, pred, target):
        return xp.mean(xp.abs(pred - target))

    def backward(self, pred, target):
        return xp.sign(pred - target)

class MAE:

    def __call__(self, prediction, target):

        return (prediction - target).abs().mean()