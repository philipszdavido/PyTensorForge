# Sequential Model

The `Sequential` model is the simplest way to build neural networks in PyTensorForge. It allows you to stack layers in order, compile the model with a loss function and optimizer, train it using `fit()`, and perform predictions.

A `Sequential` model assumes that the output of one layer is the input of the next layer.

---

# Workflow

A typical PyTensorForge workflow looks like this:

```python
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.neural.Dense import Dense

model = Sequential()

model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10))

model.compile(
    optimizer="adam",
    loss="cross_entropy_with_logits",
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
)

predictions = model.predict(X_test)
```

The training pipeline always follows the same order:

```
Create model
      ↓
Add layers
      ↓
Compile model
      ↓
Train with fit()
      ↓
Predict / Evaluate
      ↓
Save model
```

---

# Creating a Sequential Model

Create an empty model.

```python
from pytensorforge.models.seq.Sequential import Sequential

model = Sequential()
```

At this point the model contains no layers.

---

# Adding Layers

Use `add()` to append layers to the model.

```python
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10))
```

Layers are executed in the same order they are added.

For example,

```python
model = Sequential()

model.add(Embedding(...))
model.add(RNN(...))
model.add(Dense(...))
```

becomes

```
Input
  │
  ▼
Embedding
  │
  ▼
RNN
  │
  ▼
Dense
  │
  ▼
Output
```

---

# Supported Layers

Examples include:

```python
Embedding(...)
Dense(...)
RNN(...)
LSTM(...)
TransformerBlock(...)
LayerNorm(...)
Dropout(...)
LastToken(...)
```

You can freely mix compatible layers.

Example:

```python
model = Sequential()

model.add(Embedding(vocab_size, 128))
model.add(LSTM(256))
model.add(Dense(vocab_size))
```

---

# Compiling the Model

Before training, the model must be compiled.

```python
model.compile(
    optimizer="adam",
    loss="cross_entropy_with_logits",
)
```

or

```python
from pytensorforge.optimizers import Adam

model.compile(
    optimizer=Adam(lr=1e-4),
    loss="cross_entropy_with_logits",
)
```

Compilation configures:

* Optimizer
* Loss function

Compilation does **not** train the model.

---

# Available Optimizers

Examples:

```python
optimizer="sgd"
optimizer="adam"
optimizer="adamw"
```

or

```python
optimizer=Adam(lr=1e-4)
```

---

# Available Loss Functions

Regression

```python
loss="mse"
```

Classification

```python
loss="cross_entropy"
loss="cross_entropy_with_logits"
```

For classification, `cross_entropy_with_logits` is recommended because it is numerically stable and combines Softmax with Cross Entropy into a single operation.

---

# Training

Train the model using `fit()`.

```python
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
)
```

Arguments

| Argument     | Description                  |
| ------------ | ---------------------------- |
| `X_train`    | Input Tensor                 |
| `y_train`    | Target Tensor                |
| `epochs`     | Number of training passes    |
| `batch_size` | Samples processed per update |

Example

```python
model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=4,
)
```

During training PyTensorForge performs:

```
Forward Pass
        ↓
Loss Computation
        ↓
Backward Pass
        ↓
Optimizer Step
        ↓
Repeat
```

---

# Prediction

Use `predict()` after training.

```python
predictions = model.predict(X_test)
```

For language models:

```python
tokens = Tensor([[4, 2, 8]])

logits = model.predict(tokens)
```

---

# Model Summary

Display the architecture.

```python
model.summary()
```

Example

```
============================================================
Model: Sequential
============================================================
Layer (type)

Embedding
TransformerBlock
TransformerBlock
LastToken
Dense

Total params: 148,233
============================================================
```

---

# Saving a Model

Save learned parameters.

```python
model.save("model.ptf")
```

Metadata can be saved separately.

```python
model.save_metadata(
    "model.ptf",
    metadata,
)
```

or as part of a checkpoint.

---

# Loading a Model

Recreate the architecture first.

```python
model = Sequential()

model.add(...)
model.add(...)
model.add(...)
```

Compile the model.

```python
model.compile(
    optimizer="adam",
    loss="cross_entropy_with_logits",
)
```

Build the weights.

```python
dummy = Tensor(
    np.zeros((1, sequence_length)),
    requires_grad=False,
)

model(dummy)
```

Finally load the parameters.

```python
model.load("model.ptf")
```

The architecture must exactly match the saved model.

---

# Example: Feedforward Network

```python
model = Sequential()

model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
)

model.fit(
    X_train,
    y_train,
    epochs=200,
)
```

---

# Example: RNN Language Model

```python
model = Sequential()

model.add(
    Embedding(
        vocab_size=vocab_size,
        embedding_dim=128,
    )
)

model.add(
    RNN(
        hidden_size=256,
    )
)

model.add(
    Dense(vocab_size)
)

model.compile(
    optimizer="adam",
    loss="cross_entropy_with_logits",
)

model.fit(
    X_train,
    y_train,
    epochs=100,
)
```

---

# Example: Transformer Language Model

```python
model = Sequential()

model.add(
    Embedding(vocab_size, 256)
)

for _ in range(6):
    model.add(
        TransformerBlock(
            d_model=256,
            num_heads=8,
            ff_dim=1024,
        )
    )

model.add(LastToken())

model.add(
    Dense(vocab_size)
)

model.compile(
    optimizer="adam",
    loss="cross_entropy_with_logits",
)

model.fit(
    X_train,
    y_train,
    epochs=300,
)
```

---

# Complete Training Pipeline

```
Create Sequential
        │
        ▼
Add Layers
        │
        ▼
Compile
        │
        ▼
Initialize Weights (first forward pass)
        │
        ▼
Forward Propagation
        │
        ▼
Compute Loss
        │
        ▼
Backpropagation
        │
        ▼
Optimizer Update
        │
        ▼
Repeat for every Epoch
        │
        ▼
Predict / Save
```

This workflow is the recommended way to build and train neural networks in PyTensorForge. It mirrors the high-level APIs found in frameworks such as Keras while remaining lightweight and easy to extend with custom layers, loss functions, optimizers, and models.
