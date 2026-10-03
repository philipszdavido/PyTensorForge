#!/usr/bin/env python3
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile

MODEL_URL = os.environ.get("MODEL_URL", "").strip()
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/model").rstrip("/")
EXPECTED_SHA = os.environ.get("MODEL_SHA256", "").strip().lower()
TOKEN = os.environ.get("MODEL_URL_TOKEN", "").strip()
ALLOW_HTTP = os.environ.get("MODEL_ALLOW_HTTP", "") == "1"
MARKER = os.path.join(MODEL_DIR, ".source")


def log(msg):
    print(f"[fetch_model] {msg}", flush=True)


def source_id():
    return f"{MODEL_URL}|{EXPECTED_SHA}"


def already_present():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        return False
    if not MODEL_URL:
        return True
    try:
        with open(MARKER) as f:
            return f.read().strip() == source_id()
    except OSError:
        return False


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def download(url, dest):
    headers = {"User-Agent": "ptf-fetch-model/1.0"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(_DropAuthOnCrossHostRedirect())
    digest = hashlib.sha256()
    total = 0
    with opener.open(req, timeout=60) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    log(f"downloaded {total / 1e6:.1f} MB")
    return digest.hexdigest()


def safe_extract_tar(archive, dest):
    with tarfile.open(archive) as tar:
        root = os.path.realpath(dest)
        for m in tar.getmembers():
            target = os.path.realpath(os.path.join(dest, m.name))
            if not (target == root or target.startswith(root + os.sep)):
                raise ValueError(f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                raise ValueError(f"links/devices not allowed in archive: {m.name}")
        tar.extractall(dest)


def safe_extract_zip(archive, dest):
    with zipfile.ZipFile(archive) as z:
        root = os.path.realpath(dest)
        for name in z.namelist():
            target = os.path.realpath(os.path.join(dest, name))
            if not (target == root or target.startswith(root + os.sep)):
                raise ValueError(f"unsafe path in archive: {name}")
        z.extractall(dest)


def find_model_root(path):
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
            return sub
    raise ValueError("no config.json found in the archive: is this an exported model directory?")


def main():
    if already_present():
        log(f"model already present in {MODEL_DIR}")
        return 0

    if not MODEL_URL:
        log(f"no model in {MODEL_DIR} and MODEL_URL is not set")
        return 1

    if not (MODEL_URL.startswith("https://") or (ALLOW_HTTP and MODEL_URL.startswith("http://"))):
        log("MODEL_URL must be https:// (set MODEL_ALLOW_HTTP=1 only for local tests)")
        return 1

    parent = os.path.dirname(MODEL_DIR) or "."
    os.makedirs(parent, exist_ok=True)
    work = tempfile.mkdtemp(prefix="model-dl-", dir=parent)

    try:
        archive = os.path.join(work, "archive")
        log(f"downloading {MODEL_URL.split('?')[0]}")
        got = download(MODEL_URL, archive)

        if EXPECTED_SHA and got != EXPECTED_SHA:
            log(f"SHA-256 mismatch: expected {EXPECTED_SHA}, got {got}")
            return 1

        extract_to = os.path.join(work, "extracted")
        os.makedirs(extract_to)

        if zipfile.is_zipfile(archive):
            safe_extract_zip(archive, extract_to)
        elif tarfile.is_tarfile(archive):
            safe_extract_tar(archive, extract_to)
        else:
            log("archive is neither zip nor tar")
            return 1

        root = find_model_root(extract_to)

        if os.path.isdir(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

        shutil.move(root, MODEL_DIR)

        with open(MARKER, "w") as f:
            f.write(source_id())

        log(f"model ready in {MODEL_DIR}")
        return 0
    except Exception as exc:
        log(f"failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
