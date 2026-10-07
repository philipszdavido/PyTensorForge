class Loss:
    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self, pred, target):
        raise NotImplementedError