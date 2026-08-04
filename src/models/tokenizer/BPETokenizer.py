from collections import Counter

# tokenizer.fit(texts)
#
# tokenizer.encode(text)
#
# tokenizer.decode(ids)
#
# tokenizer.batch_encode(texts)
#
# tokenizer.batch_decode(batch)
#
# tokenizer.save(path)
#
# Tokenizer.load(path)
#
# len(tokenizer)
#
# tokenizer.vocab_size

# tokenizer = Tokenizer(
#     lower=True,
#     oov_token="<unk>",
# )
#
# tokenizer.fit(texts)
#
# ids = tokenizer.encode("What is Python?")
# text = tokenizer.decode(ids)
#
# batch = tokenizer.batch_encode(texts)
#
# tokenizer.save("tokenizer.ptf")
# tokenizer = Tokenizer.load("tokenizer.ptf")

class BPETokenizer:

    def __init__(self,
                 vocab_size=30000,
                 lowercase=True):

        self.vocab_size = vocab_size
        self.lowercase = lowercase

        self.merges = []
        self.word_to_index = {}
        self.index_to_word = {}


    def build_words(self, texts):

        words = Counter()

        for text in texts:

            if self.lowercase:
                text = text.lower()

            for word in text.split():

                chars = tuple(word) + ("</w>",)

                words[chars] += 1

        return words

    def get_pair_counts(self, words):

        pairs = Counter()

        for word, freq in words.items():

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])

                pairs[pair] += freq

        return pairs

    def merge(self, words, pair):

        merged = {}

        bigram = pair

        replacement = "".join(pair)

        for word, freq in words.items():

            new = []

            i = 0

            while i < len(word):

                if (
                        i < len(word) - 1
                        and word[i] == bigram[0]
                        and word[i + 1] == bigram[1]
                ):

                    new.append(replacement)

                    i += 2

                else:

                    new.append(word[i])

                    i += 1

            merged[tuple(new)] = freq

        return merged

    def fit(self, texts):

        words = self.build_words(texts)

        while True:

            pairs = self.get_pair_counts(words)

            if not pairs:
                break

            best = max(
                pairs,
                key=pairs.get
            )

            self.merges.append(best)

            words = self.merge(words, best)

            vocab = set()

            for word in words:
                vocab.update(word)

            if len(vocab) >= self.vocab_size:
                break

        vocab = sorted(vocab)

        self.word_to_index = {
            token: i
            for i, token in enumerate(vocab)
        }

        self.index_to_word = {
            i: token
            for token, i in self.word_to_index.items()
        }

    def encode_word(self, word):

        tokens = list(word)

        tokens.append("</w>")

        for pair in self.merges:

            replacement = "".join(pair)

            i = 0

            merged = []

            while i < len(tokens):

                if (
                        i < len(tokens) - 1
                        and tokens[i] == pair[0]
                        and tokens[i + 1] == pair[1]
                ):

                    merged.append(replacement)

                    i += 2

                else:

                    merged.append(tokens[i])

                    i += 1

            tokens = merged

        return [
            self.word_to_index[t]
            for t in tokens
        ]

    def encode(self, text):

        if self.lowercase:
            text = text.lower()

        ids = []

        for word in text.split():
            ids.extend(self.encode_word(word))

        return ids

    def decode(self, ids):

        tokens = [
            self.index_to_word[i]
            for i in ids
        ]

        text = ""

        for token in tokens:

            if token == "</w>":

                text += " "

            else:

                text += token

        return text.strip()


