class Hinge:

    def __call__(self, prediction, target):

        return (1 - target * prediction).maximum(0).mean()