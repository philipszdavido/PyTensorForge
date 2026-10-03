from src.core.Tensor import Tensor
from src.loss.MSELoss import MSELoss
from src.models.regression.Linear import Linear
import numpy as np

from src.optimizers.SGD import SGD


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

# 5 samples, 3 features
X = Tensor(normalize(np.array([
    [1000, 2, 10],
    [1500, 3, 5],
    [1800, 3, 15],
    [2400, 4, 2],
    [3000, 4, 20]
], dtype=np.int32)))

y = Tensor(np.array([200, 290, 310, 400, 510], dtype=np.int32))

linear = Linear([3], [1])
loss = MSELoss()
sgd = SGD(linear.parameters(), linear.grads(), learning_rate=0.01)
batch_size = 32

for epoch in range(1000):

    total_loss = 0.0
    sgd.zero_out()

    for x, target in zip(X, y):

        predicted = linear.forward(x)
        total_loss += loss.forward(predicted, target)

        # backward
        gradient = loss.backward(predicted, target)

        linear.backward(gradient)

    sgd.step()

    avg_loss = total_loss / len(X)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

for x, target in zip(X, y):
    print(linear.forward(x), target)