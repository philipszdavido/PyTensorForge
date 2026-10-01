from collections import deque

import numpy as np

from src.data.document_stream import DocumentReader
from src.data.parallel_encode import ParallelEncoder, encode_serial


class DatasetState:

    def __init__(self, file_index=0, byte_offset=0, token_buffer=None):
        self.file_index = file_index
        self.byte_offset = byte_offset
        self.token_buffer = list(token_buffer) if token_buffer else []

    def to_dict(self):
        return {
            "kind": "text",
            "file_index": self.file_index,
            "byte_offset": self.byte_offset,
            "token_buffer": list(self.token_buffer),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            file_index=data.get("file_index", 0),
            byte_offset=data.get("byte_offset", 0),
            token_buffer=data.get("token_buffer", []),
        )


class StreamingTextDataset:

    def __init__(
        self,
        corpus,
        tokenizer,
        sequence_length,
        read_buffer_size=1 << 20,
        insert_eos=True,
        text_field="text",
        state=None,
        workers=1,
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.insert_eos = insert_eos
        self.workers = max(1, int(workers))

        self.reader = DocumentReader(
            corpus.files,
            read_buffer_size=read_buffer_size,
            text_field=text_field,
        )

        self.state = state or DatasetState()
        self.token_buffer = deque(self.state.token_buffer)

    def _encoded_documents(self, doc_iter):
        if self.workers > 1:
            return ParallelEncoder(self.tokenizer, self.workers).encode(doc_iter)

        return encode_serial(self.tokenizer, doc_iter)

    def _fill_buffer(self, doc_iter, min_tokens):
        while len(self.token_buffer) < min_tokens:
            try:
                file_index, offset, ids, doc_end = next(doc_iter)
            except StopIteration:
                return False

            self.state.file_index = file_index
            self.state.byte_offset = offset

            if len(ids):
                self.token_buffer.extend(ids.tolist() if isinstance(ids, np.ndarray) else ids)

            if doc_end and self.insert_eos and self.tokenizer.eos_id is not None:
                self.token_buffer.append(self.tokenizer.eos_id)

        return True

    def batches(self, batch_size):
        needed = self.sequence_length + 1
        doc_iter = self._encoded_documents(self.reader.iter_documents(
            self.state.file_index,
            self.state.byte_offset,
        ))

        try:
            yield from self._batches(doc_iter, batch_size, needed)
        finally:
            doc_iter.close()

    def _batches(self, doc_iter, batch_size, needed):
        while True:
            examples = []

            for _ in range(batch_size):
                has_more = self._fill_buffer(doc_iter, needed)

                if len(self.token_buffer) < needed:
                    break

                block = [self.token_buffer.popleft() for _ in range(needed)]
                examples.append(block)

                if not has_more and len(self.token_buffer) < needed:
                    break

            if not examples:
                return

            arr = np.array(examples, dtype=np.int64)

            self.state.token_buffer = list(self.token_buffer)

            yield arr[:, :-1], arr[:, 1:]

    def start_new_epoch(self):
        self.state = DatasetState()
        self.token_buffer = deque()

    def state_dict(self):
        self.state.token_buffer = list(self.token_buffer)
        return self.state.to_dict()

    def load_state_dict(self, data):
        self.state = DatasetState.from_dict(data)
        self.token_buffer = deque(self.state.token_buffer)
