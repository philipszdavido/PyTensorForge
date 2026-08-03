import numpy as np
from src.loss import losses
from src.optimizers import optimizers

class Sequential:

    def __init__(self):
        self.layers = []

        self.loss = None
        self.optimizer = None
        self.metrics = []

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def compile(
        self,
        loss,
        optimizer,
        metrics=None,
    ):
        self.loss = losses[loss]
        self.optimizer = optimizers[optimizer]
        self.metrics = metrics or []

    def fit(
        self,
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=1,
    ):
        n = len(X)

        for epoch in range(epochs):

            epoch_loss = 0.0

            for start in range(0, n, batch_size):

                end = start + batch_size

                xb = X[start:end]
                yb = y[start:end]

                self.optimizer.zero_grad(self.parameters())

                predictions = self(xb)

                loss = self.loss(predictions, yb)
                epoch_loss += loss.data.item()

                loss.backward()

                self.optimizer.step(self.parameters())

                # self.optimizer.zero_grad(self.parameters())

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def summary(self):
        print("Sequential")
        print("-------------------------")

        total = 0

        for layer in self.layers:
            print(layer)

            for p in layer.parameters():
                total += np.prod(p.shape)

        print("-------------------------")
        print("Total params:", total)

    def predict(self, X):
        predictions = []
        for layer in self.layers:
            X = layer(X)
            predictions.append(X)
        return X