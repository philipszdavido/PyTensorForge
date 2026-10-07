import hashlib
import os

import numpy as np

SHARD_FORMAT_VERSION = 1


def dtype_for_vocab(vocab_size):
    return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


def file_sha256(path, block_size=1 << 20):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


class ShardWriter:

    def __init__(self, output_dir, prefix, shard_tokens, dtype):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.dtype = np.dtype(dtype)

        self._pending = []
        self._pending_count = 0
        self._shard_index = 0
        self.shards = []

        os.makedirs(output_dir, exist_ok=True)

    def write(self, token_ids):
        self._pending.append(np.asarray(token_ids, dtype=self.dtype))
        self._pending_count += len(token_ids)

        while self._pending_count >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count):
        merged = np.concatenate(self._pending) if len(self._pending) > 1 else self._pending[0]

        chunk = merged[:count]
        rest = merged[count:]

        self._pending = [rest] if rest.size else []
        self._pending_count = int(rest.size)

        name = f"{self.prefix}-{self._shard_index:05d}.bin"
        final_path = os.path.join(self.output_dir, name)
        tmp_path = final_path + ".tmp"

        with open(tmp_path, "wb") as f:
            chunk.tofile(f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)

        self.shards.append({
            "file": name,
            "token_count": int(chunk.size),
            "dtype": self.dtype.name,
            "checksum": file_sha256(final_path),
        })

        self._shard_index += 1

    def close(self):
        if self._pending_count > 0:
            self._flush(self._pending_count)

        return self.shards
