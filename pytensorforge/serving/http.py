import asyncio
import json
import logging
import time
import uuid
from http import HTTPStatus
from urllib.parse import parse_qsl, unquote, urlsplit

log = logging.getLogger("ptf.http")

MAX_REQUEST_LINE = 8192
READ_CHUNK = 65536
TOKEN_CHARS = set("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


class HTTPError(Exception):

    def __init__(self, status, message, close=True):
        super().__init__(message)
        self.status = status
        self.message = message
        self.close = close


class ClientDisconnected(Exception):
    pass


class Request:
    __slots__ = ("method", "target", "path", "query", "version", "headers", "body", "client_ip", "request_id",
                 "received_at", "info")

    def __init__(self, method, target, version, headers, client_ip):
        self.method = method
        self.target = target
        parts = urlsplit(target)
        self.path = unquote(parts.path) or "/"
        self.query = dict(parse_qsl(parts.query, keep_blank_values=True))
        self.version = version
        self.headers = headers
        self.body = b""
        self.client_ip = client_ip
        self.request_id = uuid.uuid4().hex[:16]
        self.received_at = time.perf_counter()
        self.info = {}

    def json(self):
        if not self.body:
            raise ValueError("empty body")

        return json.loads(self.body)


class Response:

    def __init__(self, status=200, body=b"", headers=None, content_type="application/json"):
        self.status = status
        self.headers = dict(headers or {})

        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")

        self.body = body

        if content_type and "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type


class StreamingResponse:

    def __init__(self, iterator, status=200, headers=None, content_type="text/event-stream; charset=utf-8",
                 on_close=None):
        self.iterator = iterator
        self.on_close = on_close
        self.status = status
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Type", content_type)
        self.headers.setdefault("Cache-Control", "no-cache")
        self.headers.setdefault("X-Accel-Buffering", "no")


class ConnectionContext:

    def __init__(self, conn):
        self._conn = conn
        self.disconnected = asyncio.Event()
        self.write_seconds = 0.0

    @property
    def client_ip(self):
        return self._conn.client_ip


class _Connection:

    def __init__(self, reader, writer, limits):
        self.reader = reader
        self.writer = writer
        self.limits = limits
        self.buf = bytearray()
        self.eof = False
        peer = writer.get_extra_info("peername")
        self.client_ip = peer[0] if isinstance(peer, tuple) else "unknown"

    async def _fill(self, timeout):
        if self.eof:
            return False

        if timeout is None:
            data = await self.reader.read(READ_CHUNK)
        else:
            data = await asyncio.wait_for(self.reader.read(READ_CHUNK), timeout)

        if not data:
            self.eof = True
            return False

        self.buf += data
        return True

    async def read_head(self, first_timeout, rest_timeout):
        timeout = first_timeout
        deadline = None

        while True:
            idx = self.buf.find(b"\r\n\r\n")

            if idx != -1:
                if idx + 4 > self.limits.max_header_bytes:
                    raise HTTPError(431, "request headers too large")

                head = bytes(self.buf[:idx])
                del self.buf[:idx + 4]
                return head

            if len(self.buf) > self.limits.max_header_bytes:
                raise HTTPError(431, "request headers too large")

            if self.buf and deadline is None:
                deadline = time.monotonic() + rest_timeout

            if deadline is not None:
                timeout = max(0.001, deadline - time.monotonic())

            try:
                got = await self._fill(timeout)
            except asyncio.TimeoutError:
                if self.buf:
                    raise HTTPError(408, "timed out reading request headers")
                return None

            if not got:
                if self.buf:
                    raise ClientDisconnected()
                return None

    async def read_exactly(self, n, timeout):
        deadline = time.monotonic() + timeout

        while len(self.buf) < n:
            left = deadline - time.monotonic()

            if left <= 0:
                raise HTTPError(408, "timed out reading request body")

            try:
                got = await self._fill(left)
            except asyncio.TimeoutError:
                raise HTTPError(408, "timed out reading request body")

            if not got:
                raise ClientDisconnected()

        data = bytes(self.buf[:n])
        del self.buf[:n]
        return data

    async def linger(self, timeout, max_bytes):
        try:
            self.writer.write_eof()
        except (OSError, RuntimeError, AttributeError):
            pass

        deadline = time.monotonic() + timeout
        discarded = 0

        while discarded < max_bytes and not self.eof:
            left = deadline - time.monotonic()

            if left <= 0:
                return

            self.buf.clear()

            try:
                data = await asyncio.wait_for(self.reader.read(READ_CHUNK), left)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                return

            if not data:
                return

            discarded += len(data)

    async def watch_disconnect(self, ctx):
        cap = self.limits.max_header_bytes + self.limits.max_body_bytes

        try:
            while not self.eof and len(self.buf) <= cap:
                await self._fill(None)
        except (ConnectionError, OSError):
            self.eof = True

        if self.eof:
            ctx.disconnected.set()

    async def write(self, data, ctx):
        if self.writer.is_closing():
            raise ClientDisconnected()

        self.writer.write(data)
        started = time.perf_counter()

        try:
            await self.writer.drain()
        except (ConnectionError, OSError) as exc:
            raise ClientDisconnected() from exc
        finally:
            if ctx is not None:
                ctx.write_seconds += time.perf_counter() - started


def _parse_head(head, client_ip):
    try:
        text = head.decode("latin-1")
    except UnicodeDecodeError:
        raise HTTPError(400, "malformed request")

    lines = text.split("\r\n")
    request_line = lines[0]

    if len(request_line) > MAX_REQUEST_LINE:
        raise HTTPError(414, "request line too long")

    parts = request_line.split(" ")

    if len(parts) != 3:
        raise HTTPError(400, "malformed request line")

    method, target, version = parts

    if version not in ("HTTP/1.1", "HTTP/1.0"):
        raise HTTPError(505, "HTTP version not supported")

    if not method or any(c not in TOKEN_CHARS for c in method):
        raise HTTPError(400, "malformed method")

    if not target.startswith("/"):
        raise HTTPError(400, "only origin-form request targets are supported")

    headers = {}

    for line in lines[1:]:
        if not line:
            continue

        if line[0] in " \t":
            raise HTTPError(400, "obsolete header folding is not allowed")

        name, sep, value = line.partition(":")

        if not sep or not name or any(c not in TOKEN_CHARS for c in name):
            raise HTTPError(400, "malformed header")

        name = name.lower()
        value = value.strip()

        if name in headers:
            if name in ("content-length", "host", "authorization", "transfer-encoding"):
                if headers[name] != value:
                    raise HTTPError(400, f"conflicting duplicate '{name}' header")
                continue

            headers[name] = headers[name] + ", " + value
        else:
            headers[name] = value

    return Request(method, target, version, headers, client_ip)


def _status_line(status):
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Unknown"

    return f"HTTP/1.1 {status} {phrase}\r\n"


def _encode_headers(status, headers):
    out = [_status_line(status)]

    for k, v in headers.items():
        v = str(v)

        if "\r" in v or "\n" in v or "\r" in k or "\n" in k:
            raise ValueError("header injection attempt")

        out.append(f"{k}: {v}\r\n")

    out.append("\r\n")
    return "".join(out).encode("latin-1")


def _error_body(status, message):
    kind = "invalid_request_error" if status < 500 else "server_error"
    return json.dumps({"error": {"message": message, "type": kind, "param": None, "code": None}}).encode()


class HTTPServer:

    def __init__(self, handler, host, port, limits, on_request_done=None, extra_headers=None):
        self.handler = handler
        self.host = host
        self.port = port
        self.limits = limits
        self.on_request_done = on_request_done
        self.extra_headers = dict(extra_headers or {})
        self._server = None
        self._connections = set()
        self._busy = set()
        self._active_requests = 0
        self._closing = False
        self._idle = asyncio.Event()
        self._idle.set()
        self.on_connection_change = None

    @property
    def sockets(self):
        return self._server.sockets if self._server else []

    @property
    def bound_port(self):
        socks = self.sockets
        return socks[0].getsockname()[1] if socks else None

    async def start(self):
        self._server = await asyncio.start_server(
            self._on_connection, self.host, self.port, limit=READ_CHUNK * 4, backlog=1024,
        )

    def _close_idle(self):
        for writer in list(self._connections):
            if writer not in self._busy:
                writer.close()

    async def shutdown(self, drain_timeout):
        self._closing = True

        if self._server is not None:
            self._server.close()

        self._close_idle()

        try:
            await asyncio.wait_for(self._idle.wait(), drain_timeout)
        except asyncio.TimeoutError:
            log.warning("shutdown drain timed out with %d active requests", self._active_requests)

        for writer in list(self._connections):
            writer.close()

        if self._server is not None:
            try:
                await asyncio.wait_for(self._server.wait_closed(), 5.0)
            except asyncio.TimeoutError:
                log.warning("some connections did not close cleanly")

    def _conn_changed(self):
        if self.on_connection_change:
            self.on_connection_change(len(self._connections))

    async def _on_connection(self, reader, writer):
        if len(self._connections) >= self.limits.max_connections or self._closing:
            try:
                body = _error_body(503, "too many connections")
                writer.write(_encode_headers(503, {
                    "Content-Type": "application/json", "Content-Length": len(body),
                    "Connection": "close", "Retry-After": "1",
                }) + body)
                await asyncio.wait_for(writer.drain(), 1.0)
            except Exception:
                pass
            finally:
                writer.close()
            return

        self._connections.add(writer)
        self._conn_changed()
        conn = _Connection(reader, writer, self.limits)

        try:
            await self._serve(conn)
        except (ClientDisconnected, ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("unhandled connection error")
        finally:
            self._connections.discard(writer)
            self._busy.discard(writer)
            self._conn_changed()

            try:
                writer.close()
            except Exception:
                pass

    async def _serve(self, conn):
        served = 0

        while not self._closing:
            first_timeout = self.limits.header_timeout_s if served == 0 else self.limits.keepalive_timeout_s

            try:
                head = await conn.read_head(first_timeout, self.limits.header_timeout_s)
            except HTTPError as exc:
                await self._send_simple(conn, exc.status, exc.message, keep_alive=False)
                return

            if head is None:
                return

            try:
                request = _parse_head(head, conn.client_ip)
                keep_alive = self._wants_keep_alive(request)
                await self._read_body(conn, request)
            except HTTPError as exc:
                await self._send_simple(conn, exc.status, exc.message, keep_alive=False)

                if exc.status == 413:
                    await conn.linger(2.0, 4 * self.limits.max_body_bytes)

                return

            served += 1

            if served >= self.limits.max_requests_per_connection:
                keep_alive = False

            keep_alive = await self._dispatch(conn, request, keep_alive and not self._closing)

            if not keep_alive:
                return

    def _wants_keep_alive(self, request):
        conn_header = request.headers.get("connection", "").lower()

        if request.version == "HTTP/1.0":
            return "keep-alive" in conn_header

        return "close" not in conn_header

    async def _read_body(self, conn, request):
        if "transfer-encoding" in request.headers:
            raise HTTPError(501, "chunked request bodies are not supported; send Content-Length")

        raw = request.headers.get("content-length")

        if raw is None:
            if request.method in ("POST", "PUT", "PATCH"):
                raise HTTPError(411, "Content-Length is required")
            return

        if not raw.isdigit():
            raise HTTPError(400, "invalid Content-Length")

        length = int(raw)

        if length > self.limits.max_body_bytes:
            raise HTTPError(413, f"request body exceeds {self.limits.max_body_bytes} bytes")

        if request.headers.get("expect", "").lower() == "100-continue" and length > len(conn.buf):
            await conn.write(b"HTTP/1.1 100 Continue\r\n\r\n", None)

        request.body = await conn.read_exactly(length, self.limits.body_timeout_s)

    async def _send_simple(self, conn, status, message, keep_alive):
        body = _error_body(status, message)
        headers = {"Content-Type": "application/json", "Content-Length": len(body),
                   "Connection": "keep-alive" if keep_alive else "close"}
        headers.update(self.extra_headers)

        try:
            await conn.write(_encode_headers(status, headers) + body, None)
        except ClientDisconnected:
            pass

    async def _dispatch(self, conn, request, keep_alive):
        ctx = ConnectionContext(conn)
        self._busy.add(conn.writer)
        watcher = asyncio.ensure_future(conn.watch_disconnect(ctx))
        self._active_requests += 1
        self._idle.clear()
        status = 500
        streamed = False

        try:
            try:
                response = await self.handler(request, ctx)
            except ClientDisconnected:
                return False

            status = response.status
            headers = dict(self.extra_headers)
            headers.update(response.headers)
            headers["X-Request-Id"] = request.request_id

            if isinstance(response, StreamingResponse):
                streamed = True
                return await self._write_stream(conn, request, response, headers, ctx, keep_alive)

            body = b"" if request.method == "HEAD" else response.body
            headers["Content-Length"] = len(response.body)
            headers["Connection"] = "keep-alive" if keep_alive else "close"
            await conn.write(_encode_headers(response.status, headers) + body, ctx)

            return keep_alive and not conn.eof
        except ClientDisconnected:
            return False
        finally:
            if not watcher.done():
                watcher.cancel()

                try:
                    await watcher
                except (asyncio.CancelledError, Exception):
                    pass

            self._active_requests -= 1
            self._busy.discard(conn.writer)

            if self._active_requests == 0:
                self._idle.set()

            if self.on_request_done is not None:
                try:
                    self.on_request_done(request, status, time.perf_counter() - request.received_at,
                                         ctx.write_seconds, ctx.disconnected.is_set(), streamed)
                except Exception:
                    log.exception("request completion hook failed")

    async def _write_stream(self, conn, request, response, headers, ctx, keep_alive):
        chunked = request.version == "HTTP/1.1"

        if chunked:
            headers["Transfer-Encoding"] = "chunked"
            headers["Connection"] = "keep-alive" if keep_alive else "close"
        else:
            headers["Connection"] = "close"
            keep_alive = False

        iterator = response.iterator
        completed = False

        try:
            await conn.write(_encode_headers(response.status, headers), ctx)

            async for piece in iterator:
                if ctx.disconnected.is_set():
                    raise ClientDisconnected()

                if not piece:
                    continue

                if isinstance(piece, str):
                    piece = piece.encode("utf-8")

                frame = b"%x\r\n%s\r\n" % (len(piece), piece) if chunked else piece
                await conn.write(frame, ctx)

            if chunked:
                await conn.write(b"0\r\n\r\n", ctx)

            completed = True
        finally:
            if not completed:
                ctx.disconnected.set()

            try:
                await iterator.aclose()
            finally:
                if response.on_close is not None:
                    response.on_close()

        return keep_alive and not conn.eof
