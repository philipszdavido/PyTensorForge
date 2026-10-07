from pytensorforge.core.Tensor import Tensor
from pytensorforge.models.embedding.Embedding import Embedding
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.neural.Dense import Dense
from pytensorforge.neural.RNN import RNN

text = """
the cat sat on the mat
the dog sat on the rug
the cat chased the mouse
the dog chased the cat
"""

####################################################
# Build vocabulary
####################################################

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

####################################################
# Dataset
####################################################

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

X_train = Tensor(X, requires_grad=False)
y_train = Tensor(y, requires_grad=False)

####################################################
# Model
####################################################

model = Sequential()

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

model.summary()

####################################################
# Train
####################################################

model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=4,
)

####################################################
# Save model
####################################################

model.save(
    "predict.ptf",
    metadata={
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "sequence_length": sequence_length,
        "vocab_size": len(vocab),
        "embedding_dim": 128,
        "hidden_size": 64,
    },
)

print("Model saved.")