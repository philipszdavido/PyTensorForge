class CategoricalCrossEntropy:
    def __call__(self, pred, target):
        pred = pred.argmax(-1)