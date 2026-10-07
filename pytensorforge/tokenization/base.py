from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @property
    @abstractmethod
    def vocab_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def bos_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_id(self):
        raise NotImplementedError

    @property
    def pad_id(self):
        return None

    @property
    def special_tokens(self):
        return {}

    def token_to_id(self, token):
        return self.special_tokens.get(token)

    @property
    @abstractmethod
    def identity(self):
        raise NotImplementedError

    @abstractmethod
    def encode(self, text, add_bos=False, add_eos=False):
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids):
        raise NotImplementedError

    def batch_encode(self, texts, add_bos=False, add_eos=False):
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos)
            for text in texts
        ]

    def batch_decode(self, batches):
        return [self.decode(ids) for ids in batches]

    def encode_stream(self, texts, add_bos=False, add_eos=False):
        for text in texts:
            for token_id in self.encode(text, add_bos=add_bos, add_eos=add_eos):
                yield token_id

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path):
        raise NotImplementedError


def identity_matches(saved, current):
    if not saved or not current:
        return False

    shared = set(saved) & set(current)

    return "type" in shared and all(saved[k] == current[k] for k in shared)
