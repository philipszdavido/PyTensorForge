import json
import os

import numpy as np

from src.data.shard_builder import MANIFEST_NAME
from src.tokenization.base import identity_matches


def is_shard_dir(path):
    return isinstance(path, str) and os.path.isfile(os.path.join(path, MANIFEST_NAME))


class ShardIndex:

    def __init__(self, directory):
        self.directory = directory

        with open(os.path.join(directory, MANIFEST_NAME)) as f:
            self.manifest = json.load(f)

        self.files = [os.path.join(directory, s["file"]) for s in self.manifest["shards"]]

    def __len__(self):
        return len(self.files)

    def state_dict(self):
        return {
            "corpus_version": self.manifest["corpus_version"],
            "total_tokens": self.manifest["total_tokens"],
            "shards": [s["file"] for s in self.manifest["shards"]],
        }

    def load_state_dict(self, state):
        if state.get("shards") != [s["file"] for s in self.manifest["shards"]] or \
                state.get("corpus_version") != self.manifest["corpus_version"]:
            raise ValueError("token shard set does not match the checkpoint; resuming would corrupt training")


class ShardedTokenDataset:

    def __init__(self, directory, sequence_length, tokenizer=None):
        self.corpus = ShardIndex(directory)
        self.sequence_length = sequence_length
        self.dtype = np.dtype(self.corpus.manifest["dtype"])

        if tokenizer is not None and not identity_matches(self.corpus.manifest["tokenizer_identity"], tokenizer.identity):
            raise ValueError("token shards were built with a different tokenizer than the one supplied")

        self.shard_index = 0
        self.token_offset = 0
        self._carry = np.empty(0, dtype=np.int64)

    def batches(self, batch_size):
        needed = self.sequence_length + 1
        meta = self.corpus.manifest["shards"]
        read_tokens = max(needed * batch_size * 8, 1 << 16)

        while True:
            examples = []

            while len(examples) < batch_size:
                while self._carry.size < needed and self.shard_index < len(meta):
                    path = self.corpus.files[self.shard_index]
                    count = meta[self.shard_index]["token_count"]

                    if self.token_offset >= count:
                        self.shard_index += 1
                        self.token_offset = 0
                        continue

                    take = min(read_tokens, count - self.token_offset)
                    chunk = np.fromfile(
                        path,
                        dtype=self.dtype,
                        count=take,
                        offset=self.token_offset * self.dtype.itemsize,
                    ).astype(np.int64)

                    self.token_offset += take
                    self._carry = np.concatenate([self._carry, chunk]) if self._carry.size else chunk

                if self._carry.size < needed:
                    break

                examples.append(self._carry[:needed])
                self._carry = self._carry[needed:]

            if not examples:
                return

            arr = np.stack(examples)

            yield arr[:, :-1], arr[:, 1:]

    def start_new_epoch(self):
        self.shard_index = 0
        self.token_offset = 0
        self._carry = np.empty(0, dtype=np.int64)

    def state_dict(self):
        return {
            "kind": "shards",
            "shard_index": self.shard_index,
            "token_offset": self.token_offset,
            "carry": self._carry.tolist(),
        }

    def load_state_dict(self, data):
        self.shard_index = data["shard_index"]
        self.token_offset = data["token_offset"]
        self._carry = np.asarray(data["carry"], dtype=np.int64)
