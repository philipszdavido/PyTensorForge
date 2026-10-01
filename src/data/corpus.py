import os
import random


class CorpusIndex:

    def __init__(
        self,
        paths,
        extensions=(".txt", ".jsonl", ".json"),
        recursive=True,
        shuffle=False,
        seed=0,
    ):
        self.extensions = tuple(e.lower() for e in extensions)
        self.recursive = recursive
        self.shuffle = shuffle
        self.seed = seed

        self.files = self._discover(paths)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.files)

    def _discover(self, paths):
        if isinstance(paths, str):
            paths = [paths]

        found = []

        for path in paths:
            path = os.path.abspath(path)

            if os.path.isfile(path):
                if path.lower().endswith(self.extensions):
                    found.append(path)
                continue

            if not os.path.isdir(path):
                raise FileNotFoundError(f"corpus path not found: {path}")

            if self.recursive:
                for root, _, names in os.walk(path):
                    for name in sorted(names):
                        if name.lower().endswith(self.extensions):
                            found.append(os.path.join(root, name))
            else:
                for name in sorted(os.listdir(path)):
                    full = os.path.join(path, name)
                    if os.path.isfile(full) and name.lower().endswith(self.extensions):
                        found.append(full)

        found.sort()

        return found

    def __len__(self):
        return len(self.files)

    def state_dict(self):
        return {
            "files": list(self.files),
            "shuffle": self.shuffle,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        if state["files"] != self.files:
            raise ValueError(
                "checkpoint corpus file list does not match the current "
                "corpus; resuming would silently skip or duplicate data"
            )

    def split(self, validation_fraction, seed=0):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")

        import hashlib

        train, val = [], []

        for path in self.files:
            h = hashlib.sha256(f"{seed}:{os.path.basename(path)}".encode()).digest()
            bucket = int.from_bytes(h[:8], "big") / 2**64
            (val if bucket < validation_fraction else train).append(path)

        if not val and len(train) > 1:
            val.append(train.pop())

        if not train:
            raise ValueError("split left no training files; use a smaller validation_fraction or more files")

        a = CorpusIndex.__new__(CorpusIndex)
        b = CorpusIndex.__new__(CorpusIndex)

        for obj, files in ((a, train), (b, val)):
            obj.extensions = self.extensions
            obj.recursive = self.recursive
            obj.shuffle = self.shuffle
            obj.seed = self.seed
            obj.files = files

        return a, b
