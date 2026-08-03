class Huber:

    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, prediction, target):

        error = prediction - target

        abs_error = error.abs()

        quadratic = 0.5 * error * error

        linear = self.delta * (
            abs_error - 0.5 * self.delta
        )

        return abs_error.where(
            abs_error < self.delta,
            quadratic,
            linear,
        ).mean()