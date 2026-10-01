import hashlib
import heapq
import json
import os
import re
from collections import Counter

from src.tokenization.base import Tokenizer

FORMAT_VERSION = 1
BASE_VOCAB = 256
HEAP_PATH_THRESHOLD = 128
CACHE_LIMIT = 1_000_000

PATTERNS = {
    "gpt2": re.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
    ),
}


def _bpe_simple(ids, ranks):
    while len(ids) > 1:
        best_rank = None

        for i in range(len(ids) - 1):
            rank = ranks.get((ids[i], ids[i + 1]))

            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best = (ids[i], ids[i + 1])

        if best_rank is None:
            break

        new_id = BASE_VOCAB + best_rank
        merged = []
        i = 0
        n = len(ids)

        while i < n:
            if i < n - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        ids = merged

    return ids


def _bpe_heap(ids, ranks):
    n = len(ids)
    ids = list(ids)
    nxt = list(range(1, n + 1))
    prv = list(range(-1, n - 1))
    alive = [True] * n
    heap = []

    for i in range(n - 1):
        rank = ranks.get((ids[i], ids[i + 1]))
        if rank is not None:
            heap.append((rank, i))

    heapq.heapify(heap)

    while heap:
        rank, i = heapq.heappop(heap)

        if not alive[i]:
            continue

        j = nxt[i]

        if j >= n or not alive[j]:
            continue

        if ranks.get((ids[i], ids[j])) != rank:
            continue

        ids[i] = BASE_VOCAB + rank
        alive[j] = False
        after = nxt[j]
        nxt[i] = after

        if after < n:
            prv[after] = i

        before = prv[i]

        if before >= 0:
            r = ranks.get((ids[before], ids[i]))
            if r is not None:
                heapq.heappush(heap, (r, before))

        if after < n:
            r = ranks.get((ids[i], ids[after]))
            if r is not None:
                heapq.heappush(heap, (r, i))

    return [ids[i] for i in range(n) if alive[i]]


class ByteLevelBPETokenizer(Tokenizer):

    def __init__(
        self,
        merges=(),
        special_tokens=None,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        pattern="gpt2",
    ):
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pretokenizer pattern '{pattern}'")

        self.pattern_name = pattern
        self._pat = PATTERNS[pattern]

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        names = [bos_token, eos_token, pad_token]

        for token in special_tokens or []:
            if token not in names:
                names.append(token)

        if len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError("special tokens must be unique and non-empty")

        self._special_names = names
        self._set_merges([tuple(m) for m in merges])

    def _set_merges(self, merges):
        ranks = {}
        token_bytes = [bytes([i]) for i in range(BASE_VOCAB)]

        for rank, (a, b) in enumerate(merges):
            if not (0 <= a < len(token_bytes) and 0 <= b < len(token_bytes)):
                raise ValueError(f"merge {rank} references an id that does not exist yet")

            if (a, b) in ranks:
                raise ValueError(f"duplicate merge {(a, b)}")

            ranks[(a, b)] = rank
            token_bytes.append(token_bytes[a] + token_bytes[b])

        self.merges = merges
        self._ranks = ranks
        self._token_bytes = token_bytes
        self._cache = {}

        base = len(token_bytes)
        self._special_ids = {name: base + i for i, name in enumerate(self._special_names)}
        self._id_to_special = {v: k for k, v in self._special_ids.items()}

        ordered = sorted(self._special_names, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(n) for n in ordered))
        self._fingerprint_value = None

    @property
    def vocab_size(self):
        return len(self._token_bytes) + len(self._special_names)

    @property
    def bos_id(self):
        return self._special_ids[self.bos_token]

    @property
    def eos_id(self):
        return self._special_ids[self.eos_token]

    @property
    def pad_id(self):
        return self._special_ids[self.pad_token]

    @property
    def special_tokens(self):
        return dict(self._special_ids)

    @property
    def fingerprint(self):
        if self._fingerprint_value is None:
            digest = hashlib.sha256()
            digest.update(json.dumps(self.merges).encode())
            digest.update(json.dumps(self._special_names).encode())
            digest.update(self.pattern_name.encode())
            self._fingerprint_value = digest.hexdigest()[:16]

        return self._fingerprint_value

    @property
    def identity(self):
        return {
            "type": "bytebpe",
            "format_version": FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "fingerprint": self.fingerprint,
        }

    def token_bytes(self, token_id):
        if 0 <= token_id < len(self._token_bytes):
            return self._token_bytes[token_id]

        if token_id in self._id_to_special:
            return b""

        raise ValueError(f"token id {token_id} is outside the vocabulary of {self.vocab_size}")

    def _encode_piece(self, piece):
        cached = self._cache.get(piece)

        if cached is not None:
            return cached

        ids = list(piece.encode("utf-8", errors="replace"))

        if len(ids) > 1:
            ids = _bpe_heap(ids, self._ranks) if len(ids) > HEAP_PATH_THRESHOLD else _bpe_simple(ids, self._ranks)

        result = tuple(ids)

        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()

        self._cache[piece] = result

        return result

    def encode_ordinary(self, text):
        out = []

        for piece in self._pat.findall(text):
            out.extend(self._encode_piece(piece))

        return out

    def encode(self, text, add_bos=False, add_eos=False, allow_special=False):
        out = [self.bos_id] if add_bos else []

        if allow_special:
            pos = 0

            for match in self._special_re.finditer(text):
                if match.start() > pos:
                    out.extend(self.encode_ordinary(text[pos:match.start()]))

                out.append(self._special_ids[match.group()])
                pos = match.end()

            if pos < len(text):
                out.extend(self.encode_ordinary(text[pos:]))
        else:
            out.extend(self.encode_ordinary(text))

        if add_eos:
            out.append(self.eos_id)

        return out

    def decode_bytes(self, ids, skip_special=True):
        parts = []

        for i in ids:
            i = int(i)

            if i in self._id_to_special:
                if not skip_special:
                    parts.append(self._id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.token_bytes(i))

        return b"".join(parts)

    def decode(self, ids, skip_special=True):
        return self.decode_bytes(ids, skip_special).decode("utf-8", errors="replace")

    def save(self, path):
        state = {
            "type": "bytebpe",
            "format_version": FORMAT_VERSION,
            "pattern": self.pattern_name,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "special_tokens": self._special_names,
            "merges": [list(m) for m in self.merges],
            "vocab_size": self.vocab_size,
            "fingerprint": self.fingerprint,
        }

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(state, f)

        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)

        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported byte-level tokenizer format {state.get('format_version')}")

        tok = cls(
            merges=state["merges"],
            special_tokens=state["special_tokens"],
            bos_token=state["bos_token"],
            eos_token=state["eos_token"],
            pad_token=state["pad_token"],
            pattern=state["pattern"],
        )

        if state.get("fingerprint") and state["fingerprint"] != tok.fingerprint:
            raise ValueError("tokenizer file failed its integrity check")

        return tok


def count_pretokens(texts, pattern="gpt2", max_unique=1_000_000):
    pat = PATTERNS[pattern]
    counts = Counter()
    threshold = 1

    for text in texts:
        counts.update(pat.findall(text))

        if len(counts) > max_unique:
            target = int(max_unique * 0.75)

            while len(counts) > target:
                counts = Counter({k: v for k, v in counts.items() if v > threshold})
                threshold += 1

    return counts


def learn_merges(counts, num_merges, min_frequency=2, progress=None):
    merged = {}

    for piece, count in counts.items():
        key = piece.encode("utf-8", errors="replace")
        merged[key] = merged.get(key, 0) + count

    words = [list(k) for k in merged]
    freqs = list(merged.values())

    pair_counts = {}
    pair_index = {}

    for wi, word in enumerate(words):
        f = freqs[wi]

        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + f
            pair_index.setdefault(pair, set()).add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []

    for step in range(num_merges):
        best = None

        while heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)

            if current <= 0:
                continue

            if current != -neg:
                heapq.heappush(heap, (-current, pair))
                continue

            best = pair
            break

        if best is None or pair_counts[best] < min_frequency:
            break

        a, b = best
        new_id = BASE_VOCAB + step
        merges.append(best)
        touched = set()

        for wi in list(pair_index.pop(best, ())):
            word = words[wi]

            if not any(word[i] == a and word[i + 1] == b for i in range(len(word) - 1)):
                continue

            f = freqs[wi]

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= f

            new_word = []
            i = 0
            n = len(word)

            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + f
                pair_index.setdefault(pair, set()).add(wi)
                touched.add(pair)

            words[wi] = new_word

        pair_counts.pop(best, None)

        for pair in touched:
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, (-c, pair))

        if progress is not None and (step + 1) % 500 == 0:
            progress(step + 1, num_merges)

    return merges


def train_byte_bpe(
    texts,
    vocab_size,
    special_tokens=None,
    min_frequency=2,
    max_unique_words=1_000_000,
    pattern="gpt2",
    bos_token="<bos>",
    eos_token="<eos>",
    pad_token="<pad>",
    progress=None,
):
    shell = ByteLevelBPETokenizer(
        special_tokens=special_tokens,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        pattern=pattern,
    )

    num_merges = vocab_size - BASE_VOCAB - len(shell.special_tokens)

    if num_merges < 0:
        raise ValueError(
            f"vocab_size {vocab_size} is too small: {BASE_VOCAB} byte tokens plus "
            f"{len(shell.special_tokens)} special tokens are always present"
        )

    counts = count_pretokens(texts, pattern, max_unique_words)
    merges = learn_merges(counts, num_merges, min_frequency, progress)

    return ByteLevelBPETokenizer(
        merges=merges,
        special_tokens=special_tokens,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        pattern=pattern,
    )
