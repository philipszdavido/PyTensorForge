from pytensorforge.backend import xp
class SparseCategoricalCrossEntropy:

    def __call__(self, prediction, target):

        prediction = prediction.softmax()

        log_probs = prediction.log()

        return -log_probs[
            xp.arange(len(target)),
            target,
        ].mean()