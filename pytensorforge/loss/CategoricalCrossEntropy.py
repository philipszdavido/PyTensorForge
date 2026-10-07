class CategoricalCrossEntropy:

    def __call__(self, prediction, target):

        eps = 1e-7

        prediction = prediction.clip(eps, 1)

        return -(target * prediction.log()).sum(axis=1).mean()