import numpy as np
from pytensorforge.loss import losses
from pytensorforge.optimizers import optimizers
from pytensorforge.serialization.checkpoint import Checkpoint
from pytensorforge.serialization.modelio import ModelIO, flatten, unflatten


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
            # print(layer.__class__.__name__, x.shape)

        return x

    def compile(
        self,
        loss,
        optimizer,
        metrics=None,
    ):

        if isinstance(loss, str):
            try:
                self.loss = losses[loss]
            except KeyError:
                raise ValueError(f"Unknown loss '{loss}'")
        else:
            self.loss = loss

        if isinstance(optimizer, str):
            try:
                self.optimizer = optimizers[optimizer]
            except KeyError:
                raise ValueError(f"Unknown optimizer '{optimizer}'")
        else:
            self.optimizer = optimizer

        self.metrics = metrics or []

    def fit(
        self,
        X,
        y,
        epochs=5,
        initial_epoch=0,
        batch_size=32,
        verbose=1,
        checkpoint_path=None,
        checkpoint_every=1,
    ):
        n = len(X)

        for epoch in range(initial_epoch, epochs):

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

            if verbose:
                # if epoch % 20 == 0 and self.layers[1].__class__.__name__:
                #     print(np.linalg.norm(self.layers[1].attn.Wq.grad))
                #     print(np.linalg.norm(self.layers[1].attn.Wk.grad))
                #     print(np.linalg.norm(self.layers[1].attn.Wv.grad))
                #     print(np.linalg.norm(self.layers[1].attn.Wo.grad))

                print(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

            if checkpoint_path is not None:
                if (epoch + 1) % checkpoint_every == 0:
                    Checkpoint.save(
                        model=self,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        path=checkpoint_path,
                    )

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

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
                state.get(f"layer_{i}", {})
            )

    def save(self, path, metadata = None):
        ModelIO.save(self, path, metadata)

    def load_model(self, path, input_shape=None):

        if not self.built:

            if input_shape is None:
                raise RuntimeError(
                    "Model must be built before loading."
                )

            self.build(input_shape)

        return ModelIO.load(self, path)

    def load(self, path):
        return ModelIO.load(self, path)

    def save_metadata(self, path, metadata):
        ModelIO.save_data(metadata, path)

    def load_metadata(self, path):
        return ModelIO.load(path)

    def save_checkpoint(
            self,
            path,
            optimizer,
            epoch,
            loss,
            metadata=None,
    ):

        checkpoint = {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "metadata": metadata or {},
        }

        np.savez_compressed(
            path,
            **flatten(checkpoint)
        )

    def load_checkpoint(
            self,
            path,
            optimizer=None,
    ):

        data = np.load(path, allow_pickle=True)

        flat = {
            k: data[k]
            for k in data.files
        }

        checkpoint = unflatten(flat)

        self.load_state_dict(
            checkpoint["model"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        return {
            "epoch": int(checkpoint["epoch"]),
            "loss": float(checkpoint["loss"]),
            "metadata": checkpoint.get("metadata", {}),
        }

    def build(self, input_shape):

        shape = input_shape

        for layer in self.layers:
            layer.build(shape)
            layer.built = True

            shape = layer.compute_output_shape(shape)

        self.built = True