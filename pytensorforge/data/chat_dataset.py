import json
from collections import deque

import numpy as np

PACKING_MODES = ("pack", "pad")


class ChatRecordError(ValueError):
    pass


class ChatSFTDataset:

    def __init__(self, corpus, template, sequence_length, packing="pack", messages_field="messages", state=None):
        if packing not in PACKING_MODES:
            raise ValueError(f"unknown packing '{packing}'; choose one of {PACKING_MODES}")

        bad = [f for f in corpus.files if not f.lower().endswith(".jsonl")]

        if bad:
            raise ValueError(f"chat datasets must be .jsonl files; got {bad[0]}")

        self.corpus = corpus
        self.template = template
        self.tokenizer = template.tokenizer
        self.sequence_length = int(sequence_length)
        self.packing = packing
        self.messages_field = messages_field
        self.roles = set(template.template.roles)
        self.pad_id = self.tokenizer.eos_id if self.tokenizer.eos_id is not None else 0
        self.load_state_dict(state or {})

    def _records(self):
        for file_index in range(self.file_index, len(self.corpus.files)):
            path = self.corpus.files[file_index]
            offset = self.byte_offset if file_index == self.file_index else 0

            with open(path, "rb") as f:
                f.seek(offset)

                while True:
                    raw = f.readline()

                    if not raw:
                        break

                    end = f.tell()
                    line = raw.strip()

                    if line:
                        yield file_index, end, self._parse(line, path, end)
                    else:
                        yield file_index, end, None

    def _parse(self, line, path, end):
        where = f"{path} (record ending at byte {end})"

        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChatRecordError(f"{where}: invalid JSON: {exc}") from None

        messages = record.get(self.messages_field) if isinstance(record, dict) else None

        if not isinstance(messages, list) or not messages:
            raise ChatRecordError(f"{where}: expected a non-empty '{self.messages_field}' list")

        for i, m in enumerate(messages):
            if not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                raise ChatRecordError(f"{where}: message {i} needs string 'role' and 'content'")

            if m["role"] not in self.roles:
                raise ChatRecordError(
                    f"{where}: message {i} has role '{m['role']}', which the chat template does not define"
                )

        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def _conversations(self):
        for file_index, end, messages in self._records():
            self.file_index = file_index
            self.byte_offset = end

            if messages is None:
                continue

            ids, learn = self.template.training_tokens(messages)

            if not any(learn):
                self.skipped += 1
                continue

            self.records += 1
            yield ids, learn

    def _pack(self, conversations, needed):
        while True:
            while len(self.tokens) < needed:
                item = next(conversations, None)

                if item is None:
                    return None

                ids, learn = item
                self.tokens.extend(ids)
                self.learn.extend(learn)

            block = [self.tokens.popleft() for _ in range(needed)]
            mask = [self.learn.popleft() for _ in range(needed)]
            return block, mask

    def _pad(self, conversations, needed):
        while True:
            item = next(conversations, None)

            if item is None:
                return None

            ids, learn = item
            ids, learn = ids[:needed], learn[:needed]

            if len(ids) < 2 or not any(learn[1:]):
                self.truncated_away += 1
                continue

            fill = needed - len(ids)
            return ids + [self.pad_id] * fill, learn + [0] * fill

    def batches(self, batch_size):
        needed = self.sequence_length + 1
        conversations = self._conversations()
        take = self._pack if self.packing == "pack" else self._pad

        try:
            while True:
                rows = []

                for _ in range(batch_size):
                    row = take(conversations, needed)

                    if row is None:
                        break

                    rows.append(row)

                if not rows:
                    return

                ids = np.array([r[0] for r in rows], dtype=np.int64)
                learn = np.array([r[1] for r in rows], dtype=np.float32)
                yield ids[:, :-1], ids[:, 1:], learn[:, 1:]
        finally:
            conversations.close()

    def start_new_epoch(self):
        self.load_state_dict({})

    def state_dict(self):
        return {
            "kind": "chat",
            "packing": self.packing,
            "file_index": self.file_index,
            "byte_offset": self.byte_offset,
            "token_buffer": list(self.tokens),
            "learn_buffer": list(self.learn),
            "records": self.records,
            "skipped": self.skipped,
            "truncated_away": self.truncated_away,
        }

    def load_state_dict(self, data):
        if data and data.get("kind") != "chat":
            raise ValueError("dataset state is not from a chat dataset")

        if data and data.get("packing", self.packing) != self.packing:
            raise ValueError("dataset state was written with a different packing mode")

        self.file_index = int(data.get("file_index", 0))
        self.byte_offset = int(data.get("byte_offset", 0))
        self.tokens = deque(data.get("token_buffer", []))
        self.learn = deque(data.get("learn_buffer", []))
        self.records = int(data.get("records", 0))
        self.skipped = int(data.get("skipped", 0))
        self.truncated_away = int(data.get("truncated_away", 0))

        if len(self.tokens) != len(self.learn):
            raise ValueError("corrupt chat dataset state: token and mask buffers differ in length")
