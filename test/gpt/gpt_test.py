import time

import numpy as np

from pytensorforge.core.Tensor import Tensor
from pytensorforge.models.embedding.Embedding import Embedding
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.models.transformers.LastToken import LastToken
from pytensorforge.models.transformers.TransformerBlock import TransformerBlock
from pytensorforge.neural.Dense import Dense
from pytensorforge.optimizers import Adam
from pytensorforge.serialization.modelio import ModelIO

####################################################
# Load metadata
####################################################

# metadata = {}

# checkpoint = ModelIO.read("predict.ptf.npz")
#
# metadata = checkpoint["metadata"]
#
# vocab_size = int(metadata["vocab_size"])
# sequence_length = int(metadata["sequence_length"])
# embedding_dim = int(metadata["embedding_dim"])
# hidden_size = metadata["hidden_size"]
# index_to_word = {
#     int(k): v
#     for k, v in metadata["index_to_word"].items()
# }

data = ModelIO.read("transformer.ptf.npz")

metadata = data["metadata"]
vocab_size = metadata["vocab_size"]

# word_to_index = metadata["word_to_index"]
# index_to_word = metadata["index_to_word"]
sequence_length = metadata["sequence_length"]
index_to_word = {
    int(k): v
    for k, v in metadata["index_to_word"].items()
}

word_to_index = {
    k: int(v)
    for k, v in metadata["word_to_index"].items()
}

model = Sequential()

####################################################
# Build architecture
####################################################

# model.add(
#     Embedding(
#         vocab_size=9,
#         embedding_dim=256,
#     )
# )
#
# for _ in range(6):
#
#     model.add(
#         TransformerBlock(
#             d_model=256,
#             num_heads=8,
#             ff_dim=1024,
#         )
#     )
#
# model.add(LastToken())
#
# model.add(
#     Dense(
#         units=9,
#         activation="softmax",
#     )
# )
#
# model.compile(
#     optimizer="adam",
#     loss="cross_entropy",
# )

model.add(
    Embedding(
        vocab_size=vocab_size,
        embedding_dim=64,
    )
)

model.add(
    TransformerBlock(
        d_model=64,
        num_heads=4,
        ff_dim=128,
    )
)

model.add(LastToken())

model.add(
    Dense(
        vocab_size,
        #activation="softmax",
    )
)

model.compile(
    optimizer=Adam(lr=1e-4),
    loss="cross_entropy_with_logits",
)

####################################################
# Build weights
####################################################

dummy = Tensor(
    np.zeros((1,3)),
    requires_grad=False,
)

model(dummy)

# model.build((1, sequence_length))

model.load_state_dict(data["model"])

####################################################
# Load model
####################################################

# metadata = model.load("transformer.ptf.npz")
#
# word_to_index = metadata["word_to_index"]
# index_to_word = metadata["index_to_word"]

####################################################
# Prediction
####################################################

def predict(seed):

    tokens = [
        word_to_index[w]
        for w in seed.lower().split()
    ]

    x = Tensor(
        np.array([tokens]),
        requires_grad=False,
    )

    probs = model.predict(x)

    idx = int(np.argmax(probs.data))

    return index_to_word[idx]

def predict_probs(seed):
    tokens = [word_to_index[w] for w in seed.lower().split()]

    x = Tensor(np.array([tokens]), requires_grad=False)

    logits = model.predict(x)

    probs = logits.softmax().data[0]

    order = np.argsort(probs)[::-1]

    for i in order:
        print(f"{index_to_word[i]:<8} {probs[i]:.4f}")

# print("the cat sat ->", predict("the cat sat"))
# print("cat sat on ->", predict("cat sat on"))
# print("dog sat on ->", predict("dog sat on"))
# print("the dog chased ->", predict("the dog chased"))
#
# tests = [
#     ("the cat sat", "on"),
#     ("cat sat on", "the"),
#     ("sat on the", "mat"),
#     ("the dog sat", "on"),
#     ("dog sat on", "the"),
#     ("sat on the", "rug"),   # ambiguous
#     ("the cat chased", "the"),
#     ("cat chased the", "mouse"),
#     ("the dog chased", "the"),
#     ("dog chased the", "cat"),
# ]
#
# for seed, expected in tests:
#     pred = predict_probs(seed)
#     print(f"{seed:15} -> {pred:6} (expected {expected})")

def tokenize(seed):
    tokens = [word_to_index[w] for w in seed.lower().split()]
    return tokens

def detokenize(tokens):
    tokens = [index_to_word[w] for w in tokens]
    sentence = ""
    for i in tokens:
        sentence += i + " "
        time.sleep(0.2)
        print(i, end=" ")
    print("", end="\r")
    return sentence

def generate(prompt, max_tokens=50):

    eos_token = tokenize("<eos>")[0]

    tokens = tokenize(prompt)

    generated = []

    for _ in range(max_tokens):
        x = Tensor(np.array([tokens]), requires_grad=False)

        logits = model.predict(x)

        next_token = int(np.argmax(logits.data[0]))

        if next_token == eos_token:
            break

        tokens.append(next_token)
        generated.append(next_token)


    return detokenize(generated)

# generate("the cat sat")
generate("What is python?")
# generate("explain recursion")