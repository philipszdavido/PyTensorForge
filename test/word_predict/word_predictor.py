import pandas as pd

from src.core.Tensor import Tensor
from src.models.embedding.Embedding import Embedding
from src.models.seq.Sequential import Sequential
from src.neural.Dense import Dense
from src.neural.LSTM import LSTM
from src.neural.RNN import RNN
from src.serialization.modelio import ModelIO

text = """
the cat sat on the mat.
the dog sat on the rug.
the cat chased the mouse.
the dog chased the cat.
"""

words = text.lower().split()

vocab = sorted(set(words))

word_to_index = {
    word: i
    for i, word in enumerate(vocab)
}

index_to_word = {
    i: word
    for word, i in word_to_index.items()
}

sequence_length = 3

X = []
y = []

for i in range(len(words) - sequence_length):

    X.append([
        word_to_index[w]
        for w in words[i:i + sequence_length]
    ])

    y.append(
        word_to_index[
            words[i + sequence_length]
        ]
    )

X_train = Tensor(
    X,
    requires_grad=False,
)

y_train = Tensor(
    y,
    requires_grad=False,
)

model = Sequential()

# model.add(
#     Embedding(
#         vocab_size=len(vocab),
#         embedding_dim=32,
#     )
# )
model.add(
    Embedding(
        vocab_size=len(vocab),
        embedding_dim=128,
    )
)

model.add(
    RNN(
        hidden_size=64,
    )
)

# model.add(
#     LSTM(
#         hidden_size=256,
#     )
# )

model.add(
    Dense(
        units=len(vocab),
        activation="softmax",
    )
)

model.compile(
    optimizer="adam",
    loss="cross_entropy",
)

model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=4,
)

model.summary()

model.save("predict.ptf")