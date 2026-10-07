from pytensorforge.neural.Dense import Dense


class Linear(Dense):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(
            units=out_features,
            activation=None,
        )

        self.in_features = in_features
        self.use_bias = bias