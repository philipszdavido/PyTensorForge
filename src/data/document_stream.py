import json
import os

JSON_EXTENSIONS = (".jsonl", ".json")
ASCII_WS = frozenset(b" \t\n\r\x0b\x0c")
WS_BYTES = (b" ", b"\t", b"\n", b"\r", b"\x0b", b"\x0c")


def _complete_utf8_len(buf):
    n = len(buf)
    i = n - 1
    back = 0

    while i >= 0 and back < 4 and (buf[i] & 0xC0) == 0x80:
        i -= 1
        back += 1

    if i < 0:
        return n

    lead = buf[i]

    if lead < 0x80:
        need = 1
    elif lead >> 5 == 0b110:
        need = 2
    elif lead >> 4 == 0b1110:
        need = 3
    elif lead >> 3 == 0b11110:
        need = 4
    else:
        need = 1

    return n if n - i >= need else i


def _prev_char_is_space(buf, end):
    k = end - 1

    while k > 0 and (buf[k] & 0xC0) == 0x80:
        k -= 1

    try:
        return buf[k:end].decode("utf-8").isspace()
    except UnicodeDecodeError:
        return False


def _find_cut(buf):
    limit = len(buf)

    while True:
        last = max(buf.rfind(c, 0, limit) for c in WS_BYTES)

        if last <= 0:
            return -1

        start = last

        while start > 0 and buf[start - 1] in ASCII_WS:
            start -= 1

        if start == 0:
            return -1

        if not _prev_char_is_space(buf, start):
            return start

        limit = start


class DocumentReader:

    def __init__(self, files, read_buffer_size=1 << 20, text_field="text"):
        self.files = files
        self.read_buffer_size = read_buffer_size
        self.text_field = text_field

    def _read_txt(self, f, start_offset):
        buf = b""
        offset = start_offset

        while True:
            chunk = f.read(self.read_buffer_size)

            if not chunk:
                if buf:
                    offset += len(buf)
                    yield buf.decode("utf-8", errors="ignore"), offset, True
                return

            buf += chunk

            cut = _find_cut(buf)

            if cut <= 0:
                if len(buf) < 4 * self.read_buffer_size:
                    continue

                cut = _complete_utf8_len(buf)

                if cut >= len(buf):
                    cut = _complete_utf8_len(buf[:-1])

                if cut <= 0:
                    continue

            segment = buf[:cut]
            buf = buf[cut:]
            offset += len(segment)

            text = segment.decode("utf-8", errors="ignore")

            if text:
                yield text, offset, False

    def _read_jsonl(self, f, start_offset):
        buf = b""
        offset = start_offset

        while True:
            chunk = f.read(self.read_buffer_size)

            if not chunk:
                if buf.strip():
                    text = self._extract_json(buf)
                    offset += len(buf)
                    if text:
                        yield text, offset, True
                break

            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                offset += len(line) + 1

                if line.strip():
                    text = self._extract_json(line)
                    if text:
                        yield text, offset, True

    def _extract_json(self, raw_line):
        try:
            obj = json.loads(raw_line.decode("utf-8", errors="ignore"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if isinstance(obj, dict):
            value = obj.get(self.text_field, "")
            return value if isinstance(value, str) else None

        return None

    def read_file(self, path, start_offset=0):
        ext = os.path.splitext(path)[1].lower()

        with open(path, "rb") as f:
            if start_offset:
                f.seek(start_offset)

            if ext in JSON_EXTENSIONS:
                yield from self._read_jsonl(f, start_offset)
            else:
                yield from self._read_txt(f, start_offset)

    def iter_documents(self, start_file_index=0, start_byte_offset=0):
        file_index = start_file_index
        resume_offset = start_byte_offset

        while file_index < len(self.files):
            path = self.files[file_index]
            offset = resume_offset if file_index == start_file_index else 0

            for text, next_offset, doc_end in self.read_file(path, offset):
                yield file_index, next_offset, text, doc_end

            file_index += 1
