import numpy as np
from src.loss import losses
from src.optimizers import optimizers
import json
import pickle

from src.serialization.modelio import ModelIO


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

    def summary_(self):
        print("Sequential")
        print("-------------------------")

        total = 0

        for layer in self.layers:
            print(layer)

            for p in layer.parameters():
                total += np.prod(p.shape)

        print("-------------------------")
        print("Total params:", total)

    def summary(self):

        print("=" * 80)
        print(f'{"Model: Sequential":^80}')
        print("=" * 80)

        print(
            f'{"Layer (type)":30}'
            f'{"Output Shape":25}'
            f'{"Param #":>15}'
        )

        print("-" * 80)

        total = 0
        trainable = 0
        non_trainable = 0

        shape = None

        for layer in self.layers:

            params = 0

            for p in layer.parameters():
                count = int(np.prod(p.shape))
                params += count
                total += count

                if p.trainable:
                    trainable += count
                else:
                    non_trainable += count

            output_shape = (
                str(layer.output_shape)
                if hasattr(layer, "output_shape")
                else "?"
            )

            print(
                f"{layer.__class__.__name__:30}"
                f"{output_shape:25}"
                f"{params:>15,}"
            )

        print("-" * 80)

        print(f"{'Total params:':30}{total:>20,}")
        print(f"{'Trainable params:':30}{trainable:>20,}")
        print(f"{'Non-trainable params:':30}{non_trainable:>20,}")

        print("=" * 80)

        if self.optimizer is not None:
            print(f"Optimizer : {self.optimizer.__class__.__name__}")

        if self.loss is not None:
            print(f"Loss      : {self.loss.__class__.__name__}")

        if self.metrics:
            print(
                "Metrics   : "
                + ", ".join(
                    metric.__class__.__name__
                    if not isinstance(metric, str)
                    else metric
                    for metric in self.metrics
                )
            )

        print("=" * 80)

    def predict(self, X):
        predictions = []
        for layer in self.layers:
            X = layer(X)
            predictions.append(X)
        return X

    def state_dict(self):

        state = {}

        for i, layer in enumerate(self.layers):
            state[f"layer_{i}"] = layer.state_dict()

        return state

    def load_state_dict(self, state):

        for i, layer in enumerate(self.layers):
            layer.load_state_dict(
                state[f"layer_{i}"]
            )

    def save(self, path):
        ModelIO.save(self, path)

    def load(self, path):
        ModelIO.load(self, path)