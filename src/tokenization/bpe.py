import hashlib
import json

from src.models.tokenizer.BPETokenizer import BPETokenizer
from src.tokenization.base import Tokenizer

TOKENIZER_FORMAT_VERSION = 1


class PTFBPETokenizer(Tokenizer):

    def __init__(
        self,
        vocab_size=30000,
        lowercase=True,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    ):
        self._base = BPETokenizer(vocab_size=vocab_size, lowercase=lowercase)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self._bos_id = None
        self._eos_id = None
        self._pad_id = None
        self._unk_id = None

        self._fitted = False

    def fit(self, texts, max_chars=None):
        collected = []
        total_chars = 0

        for text in texts:
            collected.append(text)
            total_chars += len(text)

            if max_chars is not None and total_chars >= max_chars:
                break

        self._base.fit(collected)
        self._register_special_tokens()
        self._fitted = True

    def _register_special_tokens(self):
        next_id = len(self._base.word_to_index)

        for token in (self.bos_token, self.eos_token, self.pad_token, self.unk_token):
            if token not in self._base.word_to_index:
                self._base.word_to_index[token] = next_id
                self._base.index_to_word[next_id] = token
                next_id += 1

        self._bos_id = self._base.word_to_index[self.bos_token]
        self._eos_id = self._base.word_to_index[self.eos_token]
        self._pad_id = self._base.word_to_index[self.pad_token]
        self._unk_id = self._base.word_to_index[self.unk_token]

    @property
    def vocab_size(self):
        return len(self._base.word_to_index)

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def special_tokens(self):
        return {
            self.bos_token: self._bos_id,
            self.eos_token: self._eos_id,
            self.pad_token: self._pad_id,
            self.unk_token: self._unk_id,
        }

    def _fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps([list(m) for m in self._base.merges]).encode())
        digest.update(json.dumps(sorted(self._base.word_to_index.items())).encode())
        return digest.hexdigest()[:16]

    @property
    def identity(self):
        return {
            "type": "bpe",
            "format_version": TOKENIZER_FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "lowercase": self._base.lowercase,
            "fingerprint": self._fingerprint(),
        }

    def _encode_word_safe(self, word):
        try:
            return self._base.encode_word(word)
        except KeyError:
            return [self._unk_id] if self._unk_id is not None else []

    def encode(self, text, add_bos=False, add_eos=False):
        if self._base.lowercase:
            text = text.lower()

        ids = []

        if add_bos and self._bos_id is not None:
            ids.append(self._bos_id)

        for word in text.split():
            ids.extend(self._encode_word_safe(word))

        if add_eos and self._eos_id is not None:
            ids.append(self._eos_id)

        return ids

    def decode(self, ids):
        specials = {self._bos_id, self._eos_id, self._pad_id}

        pieces = [
            self._base.index_to_word[i]
            for i in ids
            if i in self._base.index_to_word and i not in specials
        ]

        text = "".join(pieces).replace("</w>", " ")

        return " ".join(text.split())

    def save(self, path):
        state = {
            "type": "bpe",
            "vocab_size": self._base.vocab_size,
            "lowercase": self._base.lowercase,
            "merges": [list(pair) for pair in self._base.merges],
            "word_to_index": self._base.word_to_index,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "format_version": TOKENIZER_FORMAT_VERSION,
        }

        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)

        tok = cls(
            vocab_size=state["vocab_size"],
            lowercase=state["lowercase"],
            bos_token=state["bos_token"],
            eos_token=state["eos_token"],
            pad_token=state["pad_token"],
            unk_token=state["unk_token"],
        )

        tok._base.merges = [tuple(pair) for pair in state["merges"]]
        tok._base.word_to_index = {
            k: int(v) for k, v in state["word_to_index"].items()
        }
        tok._base.index_to_word = {
            v: k for k, v in tok._base.word_to_index.items()
        }

        tok._bos_id = tok._base.word_to_index[tok.bos_token]
        tok._eos_id = tok._base.word_to_index[tok.eos_token]
        tok._pad_id = tok._base.word_to_index[tok.pad_token]
        tok._unk_id = tok._base.word_to_index[tok.unk_token]
        tok._fitted = True

        return tok
