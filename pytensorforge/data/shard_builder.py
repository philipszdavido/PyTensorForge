import hashlib
import json
import os
import time
from multiprocessing import get_context

from pytensorforge.data.document_stream import DocumentReader
from pytensorforge.data.shard_writer import SHARD_FORMAT_VERSION, ShardWriter, dtype_for_vocab
from pytensorforge.tokenization.registry import load_tokenizer

MANIFEST_NAME = "manifest.json"


def corpus_version(files):
    digest = hashlib.sha256()

    for path in files:
        stat = os.stat(path)
        digest.update(f"{path}:{stat.st_size}:{int(stat.st_mtime)}".encode())

    return digest.hexdigest()


def _partition(files, workers):
    workers = max(1, min(workers, len(files)))
    size, extra = divmod(len(files), workers)
    parts = []
    start = 0

    for i in range(workers):
        end = start + size + (1 if i < extra else 0)
        parts.append(files[start:end])
        start = end

    return parts


def _worker(args):
    worker_id, files, tokenizer_path, output_dir, shard_tokens, insert_eos, text_field, read_buffer_size = args

    tokenizer = load_tokenizer(tokenizer_path)
    writer = ShardWriter(
        output_dir,
        f"shard-w{worker_id:03d}",
        shard_tokens,
        dtype_for_vocab(tokenizer.vocab_size),
    )

    reader = DocumentReader(files, read_buffer_size=read_buffer_size, text_field=text_field)
    eos = tokenizer.eos_id

    for _, _, text, doc_end in reader.iter_documents():
        if text:
            writer.write(tokenizer.encode(text))

        if doc_end and insert_eos and eos is not None:
            writer.write([eos])

    return writer.close()


def build_shards(
    corpus,
    tokenizer_path,
    output_dir,
    shard_tokens=50_000_000,
    workers=1,
    insert_eos=True,
    text_field="text",
    read_buffer_size=1 << 20,
):
    if not corpus.files:
        raise ValueError("corpus contains no files")

    tokenizer = load_tokenizer(tokenizer_path)

    os.makedirs(output_dir, exist_ok=True)

    parts = _partition(corpus.files, workers)

    jobs = [
        (i, part, tokenizer_path, output_dir, shard_tokens, insert_eos, text_field, read_buffer_size)
        for i, part in enumerate(parts)
    ]

    started = time.time()

    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)

    shards = [shard for result in results for shard in result]

    manifest = {
        "format_version": SHARD_FORMAT_VERSION,
        "tokenizer_identity": tokenizer.identity,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype_for_vocab(tokenizer.vocab_size).name,
        "insert_eos": insert_eos,
        "corpus_version": corpus_version(corpus.files),
        "source_files": list(corpus.files),
        "total_tokens": sum(s["token_count"] for s in shards),
        "shards": shards,
        "created_at": time.time(),
        "build_seconds": time.time() - started,
    }

    tmp = os.path.join(output_dir, MANIFEST_NAME + ".tmp")

    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, os.path.join(output_dir, MANIFEST_NAME))

    return manifest
