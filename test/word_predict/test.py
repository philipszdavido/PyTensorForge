import numpy as np

from pytensorforge.core.Tensor import Tensor
from pytensorforge.models.embedding.Embedding import Embedding
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.neural.Dense import Dense
from pytensorforge.neural.RNN import RNN
from pytensorforge.serialization.modelio import ModelIO

checkpoint = ModelIO.read("predict.ptf.npz")

metadata = checkpoint["metadata"]

vocab_size = int(metadata["vocab_size"])
sequence_length = int(metadata["sequence_length"])
embedding_dim = int(metadata["embedding_dim"])
hidden_size = metadata["hidden_size"]
index_to_word = {
    int(k): v
    for k, v in metadata["index_to_word"].items()
}

word_to_index = {
    k: int(v)
    for k, v in metadata["word_to_index"].items()
}

model = Sequential()

model.add(
    Embedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
)

model.add(
    RNN(hidden_size=hidden_size)
)

model.add(
    Dense(
        units=vocab_size,
        activation="softmax",
    )
)

model.compile(
    optimizer="adam",
    loss="cross_entropy",
)

model.build((1, sequence_length))

model.load_state_dict(checkpoint["model"])


def predict(seed):

    tokens = [
        word_to_index[w]
        for w in seed.lower().split()
    ]

    x = Tensor([tokens], requires_grad=False)

    probs = model.predict(x)

    idx = int(np.argmax(probs.data))

    return index_to_word[idx]


print(predict("the cat sat"))
print(predict("the dog sat"))
print(predict("the cat chased"))