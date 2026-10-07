import json

_REGISTRY = {}


def register_tokenizer(kind, cls):
    _REGISTRY[kind] = cls


def load_tokenizer(path):
    with open(path) as f:
        kind = json.load(f).get("type", "bpe")

    if kind not in _REGISTRY:
        raise ValueError(f"no tokenizer registered for type '{kind}'")

    return _REGISTRY[kind].load(path)


def _register_builtin():
    from pytensorforge.tokenization.bpe import PTFBPETokenizer
    from pytensorforge.tokenization.bytebpe import ByteLevelBPETokenizer

    register_tokenizer("bpe", PTFBPETokenizer)
    register_tokenizer("bytebpe", ByteLevelBPETokenizer)


_register_builtin()
