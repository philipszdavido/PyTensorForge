import codecs


class StreamDecoder:

    def __init__(self, tokenizer, stop_strings=None):
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in (stop_strings or []) if s]
        self.holdback = max((len(s) for s in self.stop_strings), default=1) - 1
        self.ids = []
        self.emitted = 0
        self.text = ""
        self.stopped = False

        self._incremental = hasattr(tokenizer, "token_bytes")
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace") if self._incremental else None

    def _advance(self, token_id):
        if self._incremental:
            return self.text + self._utf8.decode(self.tokenizer.token_bytes(token_id))

        self.ids.append(token_id)
        return self.tokenizer.decode(self.ids)

    def push(self, token_id):
        if self.stopped:
            return ""

        full = self._advance(token_id)

        if not full.startswith(self.text):
            common = 0

            for a, b in zip(full, self.text):
                if a != b:
                    break
                common += 1

            self.emitted = min(self.emitted, common)

        self.text = full

        for s in self.stop_strings:
            at = full.find(s, max(self.emitted - len(s) + 1, 0))

            if at != -1:
                self.text = full[:at]
                self.stopped = True
                break

        safe = len(self.text) if self.stopped else max(len(self.text) - self.holdback, self.emitted)
        delta = self.text[self.emitted:safe]
        self.emitted = safe

        return delta

    def flush(self):
        if self._incremental and not self.stopped:
            tail = self._utf8.decode(b"", final=True)

            if tail:
                self.text += tail

        delta = self.text[self.emitted:]
        self.emitted = len(self.text)

        return delta
