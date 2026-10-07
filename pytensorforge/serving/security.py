import hashlib
import hmac
import ipaddress
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

from pytensorforge.serving import errors


def hash_key(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _parse_entry(entry):
    entry = entry.strip()

    if not entry or entry.startswith("#"):
        return None

    if entry.startswith("sha256:"):
        digest = entry[len("sha256:"):].strip().lower()

        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("malformed sha256 key entry")

        return digest

    if len(entry) < 16:
        raise ValueError("API keys must be at least 16 characters")

    return hash_key(entry)


def _read_key_file(path):
    with open(path) as f:
        return [line for line in f.read().splitlines()]


def _collect(inline, path, env_name):
    raw = list(inline or [])

    if path:
        raw.extend(_read_key_file(path))

    if env_name and os.environ.get(env_name):
        raw.extend(os.environ[env_name].split(","))

    return [d for d in (_parse_entry(e) for e in raw) if d]


@dataclass(frozen=True)
class Principal:
    id: str
    authenticated: bool
    admin: bool = False


class KeyStore:

    def __init__(self, user_digests, admin_digests, allow_unauthenticated=False):
        self._user = [bytes.fromhex(d) for d in dict.fromkeys(user_digests)]
        self._admin = [bytes.fromhex(d) for d in dict.fromkeys(admin_digests)]
        self.allow_unauthenticated = allow_unauthenticated

    @classmethod
    def from_config(cls, sec):
        users = _collect(sec.api_keys, sec.api_keys_file, sec.api_keys_env)
        admins = _collect(sec.admin_keys, sec.admin_keys_file, None)
        return cls(users, admins, sec.allow_unauthenticated)

    @property
    def enabled(self):
        return bool(self._user or self._admin)

    @staticmethod
    def extract(headers):
        auth = headers.get("authorization", "")

        if auth[:7].lower() == "bearer ":
            return auth[7:].strip() or None

        key = headers.get("x-api-key", "").strip()

        return key or None

    @staticmethod
    def _match(digest, pool):
        found = False

        for candidate in pool:
            found |= hmac.compare_digest(digest, candidate)

        return found

    def authenticate(self, headers, client_ip):
        key = self.extract(headers)

        if key is None:
            if self.enabled and not self.allow_unauthenticated:
                raise errors.unauthorized("missing API key; send 'Authorization: Bearer <key>'")

            return Principal(f"ip:{client_ip}", authenticated=False)

        digest = hashlib.sha256(key.encode("utf-8")).digest()
        is_admin = self._match(digest, self._admin)
        is_user = self._match(digest, self._user)

        if not (is_admin or is_user):
            raise errors.unauthorized()

        return Principal(f"key:{digest.hex()[:12]}", authenticated=True, admin=is_admin)


def is_loopback(host):
    if host in ("localhost", ""):
        return True

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Bucket:
    __slots__ = ("tokens", "updated", "in_flight")

    def __init__(self, capacity, now):
        self.tokens = float(capacity)
        self.updated = now
        self.in_flight = 0


class RateLimiter:

    def __init__(self, requests_per_minute, burst, max_concurrent, max_tracked=10_000, clock=time.monotonic):
        self.rate = requests_per_minute / 60.0
        self.capacity = burst
        self.max_concurrent = max_concurrent
        self.max_tracked = max_tracked
        self.clock = clock
        self._buckets = OrderedDict()

    @classmethod
    def from_config(cls, rl):
        return cls(rl.requests_per_minute, rl.burst, rl.max_concurrent_per_principal, rl.max_tracked_principals)

    def _bucket(self, key, now):
        b = self._buckets.get(key)

        if b is None:
            b = _Bucket(self.capacity, now)
            self._buckets[key] = b

            while len(self._buckets) > self.max_tracked:
                oldest, victim = next(iter(self._buckets.items()))

                if victim.in_flight:
                    self._buckets.move_to_end(oldest)
                    break

                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)

        b.tokens = min(self.capacity, b.tokens + (now - b.updated) * self.rate)
        b.updated = now

        return b

    def acquire(self, key):
        now = self.clock()
        b = self._bucket(key, now)

        if b.in_flight >= self.max_concurrent:
            raise errors.rate_limited(
                f"too many concurrent requests for this API key (limit {self.max_concurrent})", 1
            )

        if b.tokens < 1.0:
            raise errors.rate_limited("request rate limit exceeded", (1.0 - b.tokens) / self.rate)

        b.tokens -= 1.0
        b.in_flight += 1

    def release(self, key):
        b = self._buckets.get(key)

        if b is not None and b.in_flight > 0:
            b.in_flight -= 1

    def charge(self, key):
        now = self.clock()
        b = self._bucket(key, now)

        if b.tokens < 1.0:
            raise errors.rate_limited("request rate limit exceeded", (1.0 - b.tokens) / self.rate)

        b.tokens -= 1.0
