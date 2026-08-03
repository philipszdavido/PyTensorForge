import numpy as np


class SparseCategoricalCrossEntropy:

    def __call__(self, prediction, target):

        prediction = prediction.softmax()

        log_probs = prediction.log()

        return -log_probs[
            np.arange(len(target)),
            target,
        ].mean()