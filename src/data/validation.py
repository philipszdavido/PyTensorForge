import json
import os

from src.data.document_stream import JSON_EXTENSIONS
from src.data.shard_builder import MANIFEST_NAME, corpus_version
from src.data.shard_writer import file_sha256
from src.tokenization.base import identity_matches


class ValidationReport:

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}

    @property
    def ok(self):
        return not self.errors

    def to_dict(self):
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def validate_corpus(corpus, text_field="text", sample_bytes=4 << 20, tokenizer=None, sample_docs=200):
    report = ValidationReport()

    if not corpus.files:
        report.errors.append("corpus contains no supported files")
        return report

    total_bytes = 0
    malformed = 0
    bad_encoding = 0
    empty_docs = 0
    checked_docs = 0
    unk_tokens = 0
    sampled_tokens = 0
    sampled_chars = 0

    for path in corpus.files:
        if not os.path.isfile(path):
            report.errors.append(f"missing file: {path}")
            continue

        if not os.access(path, os.R_OK):
            report.errors.append(f"unreadable file: {path}")
            continue

        total_bytes += os.path.getsize(path)

    report.stats["files"] = len(corpus.files)
    report.stats["total_bytes"] = total_bytes

    remaining = sample_bytes

    for path in corpus.files:
        if remaining <= 0 or checked_docs >= sample_docs * 10:
            break

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(path)[1].lower()

        with open(path, "rb") as f:
            if ext in JSON_EXTENSIONS:
                for line in f:
                    remaining -= len(line)

                    if not line.strip():
                        continue

                    checked_docs += 1

                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError:
                        bad_encoding += 1
                        continue

                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue

                    value = obj.get(text_field) if isinstance(obj, dict) else None

                    if not isinstance(value, str):
                        malformed += 1
                    elif not value.strip():
                        empty_docs += 1
                    elif tokenizer is not None and sampled_tokens < 200000 and checked_docs <= sample_docs:
                        ids = tokenizer.encode(value)
                        sampled_tokens += len(ids)
                        sampled_chars += len(value)
                        unk = getattr(tokenizer, "unk_id", None)
                        if unk is not None:
                            unk_tokens += sum(1 for i in ids if i == unk)

                    if remaining <= 0:
                        break
            else:
                data = f.read(min(max(remaining, 0), sample_bytes))
                remaining -= len(data)
                checked_docs += 1

                try:
                    data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    if exc.start < len(data) - 4:
                        bad_encoding += 1

                if not data.strip():
                    empty_docs += 1
                elif tokenizer is not None:
                    sample_text = data.decode("utf-8", errors="ignore")[:200000]
                    ids = tokenizer.encode(sample_text)
                    sampled_tokens += len(ids)
                    sampled_chars += len(sample_text)
                    unk = getattr(tokenizer, "unk_id", None)
                    if unk is not None:
                        unk_tokens += sum(1 for i in ids if i == unk)

    report.stats.update({
        "sampled_documents": checked_docs,
        "malformed_documents": malformed,
        "encoding_errors": bad_encoding,
        "empty_documents": empty_docs,
    })

    if malformed:
        report.warnings.append(f"{malformed} malformed JSON documents in sample")

    if bad_encoding:
        report.warnings.append(f"{bad_encoding} documents with invalid UTF-8 in sample")

    if tokenizer is not None and sampled_tokens:
        report.stats["chars_per_token"] = sampled_chars / sampled_tokens

    if tokenizer is not None and sampled_tokens and getattr(tokenizer, "unk_id", None) is not None:
        rate = unk_tokens / sampled_tokens
        report.stats["unk_rate"] = rate

        if rate > 0.05:
            report.warnings.append(f"unknown-token rate {rate:.1%} in sample; tokenizer may not fit this corpus")

    return report


def validate_shards(directory, tokenizer=None, verify_checksums=False):
    report = ValidationReport()
    manifest_path = os.path.join(directory, MANIFEST_NAME)

    if not os.path.isfile(manifest_path):
        report.errors.append("manifest.json not found")
        return report

    with open(manifest_path) as f:
        manifest = json.load(f)

    itemsize = {"uint16": 2, "uint32": 4}.get(manifest["dtype"])

    if itemsize is None:
        report.errors.append(f"unsupported dtype {manifest['dtype']}")
        return report

    if tokenizer is not None:
        if not identity_matches(manifest["tokenizer_identity"], tokenizer.identity):
            report.errors.append("tokenizer identity does not match shard manifest")

        if tokenizer.vocab_size != manifest["vocab_size"]:
            report.errors.append("tokenizer vocab size does not match shard manifest")

    total = 0

    for shard in manifest["shards"]:
        path = os.path.join(directory, shard["file"])

        if not os.path.isfile(path):
            report.errors.append(f"missing shard {shard['file']}")
            continue

        size = os.path.getsize(path)

        if size != shard["token_count"] * itemsize:
            report.errors.append(f"shard {shard['file']} size does not match token_count")
            continue

        total += shard["token_count"]

        if verify_checksums and file_sha256(path) != shard["checksum"]:
            report.errors.append(f"checksum mismatch in {shard['file']}")

    if total != manifest["total_tokens"]:
        report.errors.append("manifest total_tokens does not match shard token counts")

    stale = [p for p in manifest.get("source_files", []) if not os.path.isfile(p)]

    if stale:
        report.warnings.append(f"{len(stale)} source files no longer exist; corpus_version cannot be re-verified")
    elif manifest.get("source_files") and corpus_version(manifest["source_files"]) != manifest["corpus_version"]:
        report.warnings.append("source corpus changed since shards were built")

    report.stats.update({
        "shards": len(manifest["shards"]),
        "total_tokens": total,
        "checksums_verified": verify_checksums,
    })

    return report
