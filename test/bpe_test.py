from pytensorforge.models.tokenizer.BPETokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=100)
tokenizer.fit(["the cat sat on the mat"])

ids = tokenizer.encode(
    "The cat sat"
)

print(ids)

print(tokenizer.decode(ids))