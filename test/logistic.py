from pytensorforge.loss.bce import BCELoss
from pytensorforge.loss.MSE import MSELoss
from pytensorforge.models.regression.Logistic import Logistic
import numpy as np
import matplotlib.pyplot as plt

from pytensorforge.optimizers.SGD import SGD


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

# 5 samples, 3 features
X = normalize(np.array([
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0]
], dtype=np.int32))

y = np.array([1, 1, 0, 0], dtype=np.int32)

logistic = Logistic([2], [1])
loss = MSELoss()
losses = []
sgd = SGD(logistic.parameters(), logistic.grads(), learning_rate=0.05)

for epoch in range(1000):

    total_loss = 0

    for x, target in zip(X, y):

        sgd.zero_out()

        predicted = logistic.forward(x)
        total_loss += loss.forward(predicted, target)

        grad_output = loss.backward(predicted, target)

        # backward
        logistic.backward(grad_output)

        # logistic.update(0.05)
        sgd.step()

    avg_loss = total_loss / len(X)
    losses.append(avg_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()

for x, target in zip(X, y):
    print(logistic.forward(x), target)

i = [1, 7]

print(i[-1])