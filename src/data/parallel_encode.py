import multiprocessing as mp
from collections import deque

import numpy as np

_EMPTY = np.zeros(0, dtype=np.int32)
_worker_tokenizer = None


def _init_worker(tokenizer):
    global _worker_tokenizer
    _worker_tokenizer = tokenizer


def _encode_texts(texts):
    tok = _worker_tokenizer
    return [np.asarray(tok.encode(t), dtype=np.int32) if t else _EMPTY for t in texts]


def encode_serial(tokenizer, doc_iter):
    for file_index, offset, text, doc_end in doc_iter:
        ids = tokenizer.encode(text) if text else ()
        yield file_index, offset, ids, doc_end


class ParallelEncoder:

    def __init__(self, tokenizer, workers, batch_chars=1 << 18, max_batch_docs=4096, inflight_per_worker=2,
                 start_method="spawn"):
        if workers < 2:
            raise ValueError("ParallelEncoder needs at least 2 workers; use encode_serial otherwise")

        if tokenizer.vocab_size > np.iinfo(np.int32).max:
            raise ValueError("vocabulary too large for int32 token transport")

        self.tokenizer = tokenizer
        self.workers = int(workers)
        self.batch_chars = max(1, int(batch_chars))
        self.max_batch_docs = max(1, int(max_batch_docs))
        self.max_inflight = max(1, self.workers * int(inflight_per_worker))
        self.start_method = start_method

    def _next_batch(self, it):
        meta = []
        texts = []
        chars = 0

        for file_index, offset, text, doc_end in it:
            meta.append((file_index, offset, doc_end))
            texts.append(text)
            chars += len(text)

            if chars >= self.batch_chars or len(texts) >= self.max_batch_docs:
                break

        return meta, texts

    def encode(self, doc_iter):
        ctx = mp.get_context(self.start_method)
        pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=(self.tokenizer,))
        pending = deque()
        it = iter(doc_iter)
        exhausted = False

        try:
            while True:
                while not exhausted and len(pending) < self.max_inflight:
                    meta, texts = self._next_batch(it)

                    if not meta:
                        exhausted = True
                        break

                    pending.append((meta, pool.apply_async(_encode_texts, (texts,))))

                if not pending:
                    return

                meta, result = pending.popleft()

                for (file_index, offset, doc_end), ids in zip(meta, result.get()):
                    yield file_index, offset, ids, doc_end
        finally:
            pending.clear()
            pool.terminate()
            pool.join()
