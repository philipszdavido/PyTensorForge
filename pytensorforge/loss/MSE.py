class MSE:

    def __call__(self, prediction, target):

        diff = prediction - target

        return (diff * diff).mean()