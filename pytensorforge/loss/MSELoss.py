from pytensorforge.backend import xp
from pytensorforge.loss.Loss import Loss

class MSELoss(Loss):

    def forward(self, pred, target):
        return 0.5 * xp.mean((pred - target) ** 2)

    def backward(self, pred, target):
        return pred - target
