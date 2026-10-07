import asyncio
import json
import logging
import mimetypes
import os
import re
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

from pytensorforge.inference.chat_template import ChatTemplateError, PromptTooLong
from pytensorforge.inference.scheduler import GenerationRequest
from pytensorforge.serving import errors, protocol
from pytensorforge.serving.errors import APIError
from pytensorforge.serving.http import ClientDisconnected, Response, StreamingResponse
from pytensorforge.serving.metrics import ServerMetrics
from pytensorforge.serving.security import KeyStore, RateLimiter

log = logging.getLogger("ptf.api")

MODEL_PATH_RE = re.compile(r"^/v1/models/([^/]+)$")
ADMIN_MODEL_RE = re.compile(r"^/admin/models/([^/]+)/(load|unload)$")
UI_TYPES = {".html", ".js", ".css", ".svg", ".png", ".ico", ".json", ".map", ".txt", ".woff2"}
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
UI_CSP = ("default-pytensorforge 'self'; script-pytensorforge 'self'; style-pytensorforge 'self'; img-pytensorforge 'self' data:; "
          "connect-pytensorforge 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
FINISH_TO_OPENAI = {"stop": "stop", "length": "length"}


def _json_dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def sse(obj):
    return f"data: {_json_dumps(obj)}\n\n"


class _Bridge:

    def __init__(self, handle, loop):
        self.handle = handle
        self._signal = asyncio.Event()
        self._loop = loop
        handle.set_listener(self._notify)

    def _notify(self):
        try:
            self._loop.call_soon_threadsafe(self._signal.set)
        except RuntimeError:
            pass

    async def events(self, ctx, tick_s):
        disconnect_wait = asyncio.ensure_future(ctx.disconnected.wait())

        try:
            while True:
                for ev in self.handle.drain():
                    yield ev

                    if ev.finished:
                        return

                if ctx.disconnected.is_set():
                    raise ClientDisconnected()

                self._signal.clear()
                pending = self.handle.drain()

                if pending:
                    for ev in pending:
                        yield ev

                        if ev.finished:
                            return
                    continue

                signal_wait = asyncio.ensure_future(self._signal.wait())
                done, _ = await asyncio.wait({signal_wait, disconnect_wait}, timeout=tick_s,
                                             return_when=asyncio.FIRST_COMPLETED)

                if signal_wait not in done:
                    signal_wait.cancel()

                if not done:
                    yield None
        finally:
            disconnect_wait.cancel()
            self.handle.set_listener(None)


class _LStrip:

    def __init__(self, enabled):
        self.pending = enabled

    def __call__(self, text):
        if not self.pending or not text:
            return text

        stripped = text.lstrip()

        if stripped:
            self.pending = False

        return stripped


class APIApp:

    def __init__(self, model_server, config, ui_directory=None):
        self.models = model_server
        self.config = config
        self.limits = config.limits
        self.keys = KeyStore.from_config(config.security)
        self.limiter = RateLimiter.from_config(config.security.rate_limit)
        self.metrics = ServerMetrics(weakref.ref(model_server))
        self.cors = set(config.security.cors_origins)
        self._tok_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ptf-tokenize")
        self._admin_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ptf-admin")
        self._access = self._open_access_log(config.logging.access_log)
        self._ui_files = self._index_ui(ui_directory) if config.ui.enabled else {}

    @staticmethod
    def _open_access_log(target):
        if not target:
            return None

        if target == "-":
            return sys.stderr

        return open(target, "a", buffering=1)

    @staticmethod
    def _index_ui(directory):
        if not directory or not os.path.isdir(directory):
            return {}

        files = {}
        root = os.path.realpath(directory)

        for dirpath, _, names in os.walk(root):
            for name in names:
                full = os.path.realpath(os.path.join(dirpath, name))

                if not full.startswith(root + os.sep) or os.path.splitext(name)[1] not in UI_TYPES:
                    continue

                rel = "/" + os.path.relpath(full, root).replace(os.sep, "/")
                files[rel] = full

        return files

    def close(self):
        self._tok_pool.shutdown(wait=False, cancel_futures=True)
        self._admin_pool.shutdown(wait=False, cancel_futures=True)

        if self._access not in (None, sys.stderr):
            self._access.close()

    def _client_ip(self, request):
        if self.config.security.trust_forwarded_for:
            fwd = request.headers.get("x-forwarded-for")

            if fwd:
                return fwd.split(",")[0].strip()

        return request.client_ip

    def _cors_headers(self, request):
        origin = request.headers.get("origin")

        if not origin or not self.cors:
            return {}

        if "*" in self.cors or origin in self.cors:
            return {
                "Access-Control-Allow-Origin": origin,
                "Vary": "Origin",
                "Access-Control-Expose-Headers": "X-Request-Id, Retry-After",
                "Access-Control-Max-Age": "600",
            }

        return {}

    def _error(self, request, exc):
        headers = dict(exc.headers)
        headers.update(self._cors_headers(request))
        headers.update(SECURITY_HEADERS)
        request.info.setdefault("error_code", exc.code)

        return Response(exc.status, exc.to_dict(), headers)

    def _authenticate(self, request):
        try:
            principal = self.keys.authenticate(request.headers, self._client_ip(request))
        except APIError:
            self.metrics.auth_failures.inc()
            raise

        request.info["principal"] = principal.id

        return principal

    def _route_name(self, request):
        path = request.path

        if path in ("/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/ready", "/metrics",
                    "/admin/models"):
            return path

        if MODEL_PATH_RE.match(path):
            return "/v1/models/{id}"

        if ADMIN_MODEL_RE.match(path):
            return "/admin/models/{name}/{action}"

        if path in self._ui_files or path == "/":
            return "ui"

        return "other"

    async def __call__(self, request, ctx):
        request.info["route"] = self._route_name(request)

        try:
            response = await self._handle(request, ctx)
        except ClientDisconnected:
            raise
        except APIError as exc:
            return self._error(request, exc)
        except Exception:
            log.exception("request %s failed", request.request_id)
            self.metrics.errors.inc(kind="internal")
            return self._error(request, errors.internal(request.request_id))

        response.headers.update(self._cors_headers(request))

        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)

        return response

    async def _handle(self, request, ctx):
        method, path = request.method, request.path

        if method == "OPTIONS":
            return self._preflight(request)

        if path == "/health" and method in ("GET", "HEAD"):
            return Response(200, {"status": "ok"})

        if path == "/ready" and method in ("GET", "HEAD"):
            loaded = [m.name for m in self.models.loaded_models() if m.accepting]
            return Response(200 if loaded else 503, {"status": "ready" if loaded else "not_ready", "models": loaded})

        if path == "/metrics" and method == "GET":
            if self.config.security.metrics_require_auth and self.keys.enabled:
                self._authenticate(request)

            fmt = request.query.get("format")

            if fmt == "json":
                return Response(200, self.metrics.registry.snapshot())

            return Response(200, self.metrics.registry.render_prometheus(),
                            content_type="text/plain; version=0.0.4; charset=utf-8")

        if path.startswith("/v1/"):
            return await self._handle_v1(request, ctx)

        if path.startswith("/admin/"):
            return await self._handle_admin(request)

        if method in ("GET", "HEAD") and self._ui_files:
            return self._serve_ui(request)

        raise errors.not_found(f"no route for {method} {path}", code="unknown_url")

    def _preflight(self, request):
        cors = self._cors_headers(request)

        if not cors:
            raise APIError(403, "origin not allowed", "permission_error", code="cors_forbidden")

        cors.update({
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "authorization, content-type, x-api-key",
        })

        return Response(204, b"", cors, content_type=None)

    def _serve_ui(self, request):
        path = "/index.html" if request.path in ("/", "/ui", "/ui/") else request.path
        full = self._ui_files.get(path)

        if full is None:
            raise errors.not_found("not found", code="unknown_url")

        with open(full, "rb") as f:
            body = f.read()

        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"

        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"

        headers = {"Cache-Control": "no-cache" if path.endswith(".html") else "public, max-age=300"}

        if path.endswith(".html"):
            headers["Content-Security-Policy"] = UI_CSP
            headers["X-Frame-Options"] = "DENY"

        return Response(200, body, headers, content_type=ctype)

    def _parse_json(self, request):
        ctype = request.headers.get("content-type", "application/json").split(";")[0].strip().lower()

        if ctype != "application/json":
            raise APIError(415, "Content-Type must be application/json", "invalid_request_error",
                           code="unsupported_media_type")

        try:
            return request.json()
        except (ValueError, UnicodeDecodeError):
            raise errors.bad_request("request body is not valid JSON", code="invalid_json")

    async def _handle_v1(self, request, ctx):
        method, path = request.method, request.path

        if path == "/v1/chat/completions":
            if method != "POST":
                raise APIError(405, "use POST", "invalid_request_error", code="method_not_allowed",
                               headers={"Allow": "POST"})
            return await self._completion(request, ctx, chat=True)

        if path == "/v1/completions":
            if method != "POST":
                raise APIError(405, "use POST", "invalid_request_error", code="method_not_allowed",
                               headers={"Allow": "POST"})
            return await self._completion(request, ctx, chat=False)

        if method != "GET":
            raise APIError(405, "use GET", "invalid_request_error", code="method_not_allowed",
                           headers={"Allow": "GET"})

        principal = self._authenticate(request)
        self._charge(principal)

        if path == "/v1/models":
            data = [m.describe() for m in self.models.loaded_models() if m.accepting]
            return Response(200, {"object": "list", "data": data})

        match = MODEL_PATH_RE.match(path)

        if match:
            return Response(200, self.models.get(match.group(1)).describe())

        raise errors.not_found(f"no route for {method} {path}", code="unknown_url")

    def _charge(self, principal):
        try:
            self.limiter.charge(principal.id)
        except APIError:
            self.metrics.rate_limited.inc(reason="rate")
            raise

    async def _handle_admin(self, request):
        principal = self._authenticate(request)

        if not principal.admin:
            raise errors.forbidden("admin API key required")

        if request.path == "/admin/models" and request.method == "GET":
            loaded = {m.name: m for m in self.models.loaded_models()}
            return Response(200, {"object": "list", "data": [
                {
                    "name": e.name,
                    "loaded": e.name in loaded,
                    "accepting": loaded[e.name].accepting if e.name in loaded else False,
                    "in_flight": loaded[e.name].in_flight if e.name in loaded else 0,
                    "memory_bytes": loaded[e.name].memory_bytes if e.name in loaded else None,
                }
                for e in self.models.declared()
            ]})

        match = ADMIN_MODEL_RE.match(request.path)

        if match and request.method == "POST":
            name, action = match.groups()
            loop = asyncio.get_running_loop()

            if action == "load":
                model = await loop.run_in_executor(self._admin_pool, self.models.load, name)
                return Response(200, {"status": "loaded", "model": model.describe()})

            drained = await loop.run_in_executor(
                self._admin_pool, self.models.unload, name, self.limits.shutdown_drain_s
            )
            return Response(200, {"status": "unloaded", "model": name, "drained": drained})

        raise errors.not_found(f"no route for {request.method} {request.path}", code="unknown_url")

    def _generation_config(self, model, sampling, prompt_len, chat):
        gen = model.generator.default_config
        limits = model.limits

        max_new = sampling.max_tokens or gen.max_new_tokens
        max_new = min(max_new, limits.max_generation_tokens, limits.context_length - prompt_len)

        if max_new < 1:
            raise errors.bad_request(
                f"prompt of {prompt_len} tokens leaves no room to generate within the context of "
                f"{limits.context_length} tokens",
                param="messages" if chat else "prompt", code="context_length_exceeded",
            )

        stop_strings = list(sampling.stop)
        stop_ids = list(gen.stop_token_ids)

        if chat:
            stop_strings = list(model.template.stop_strings) + stop_strings
            stop_ids = stop_ids + [i for i in model.template.stop_token_ids if i not in stop_ids]

        overrides = {
            "max_new_tokens": max_new,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "seed": sampling.seed,
            "stop_strings": stop_strings,
            "stop_token_ids": stop_ids,
        }

        if sampling.temperature is not None and sampling.temperature > 0:
            overrides["do_sample"] = True

        try:
            return gen.updated(**overrides)
        except ValueError as exc:
            raise errors.bad_request(str(exc))

    def _tokenize_chat(self, model, messages, max_new_hint):
        limits = model.limits
        reserve = min(max_new_hint, limits.context_length // 2)
        budget = min(limits.max_prompt_tokens, limits.context_length - reserve)
        truncate = self.config.runtime.chat_truncation == "auto"

        started = time.perf_counter()

        try:
            rendered = model.template.render(messages, max_prompt_tokens=budget, truncate=truncate)
        except PromptTooLong as exc:
            raise errors.bad_request(
                f"the conversation needs {exc.needed} prompt tokens but at most {exc.budget} fit "
                f"(context {limits.context_length}, {reserve} reserved for the reply)",
                param="messages", code="context_length_exceeded",
            )
        except ChatTemplateError as exc:
            raise errors.bad_request(str(exc), param="messages")

        return rendered.token_ids, rendered.messages_dropped, time.perf_counter() - started

    def _tokenize_completion(self, model, prompt):
        started = time.perf_counter()
        vocab = model.generator.config.vocab_size

        if isinstance(prompt, str):
            ids = model.generator.tokenizer.encode(prompt)
        else:
            ids = list(prompt)

            if any(t < 0 or t >= vocab for t in ids):
                raise errors.bad_request("prompt contains token ids outside the model vocabulary", param="prompt")

        if not ids:
            eos = model.generator.tokenizer.eos_id

            if eos is None:
                raise errors.bad_request("prompt must not be empty", param="prompt")

            ids = [eos]

        if len(ids) > model.limits.max_prompt_tokens:
            raise errors.bad_request(
                f"prompt of {len(ids)} tokens exceeds the limit of {model.limits.max_prompt_tokens}",
                param="prompt", code="context_length_exceeded",
            )

        return ids, 0, time.perf_counter() - started

    async def _completion(self, request, ctx, chat):
        principal = self._authenticate(request)
        body = self._parse_json(request)
        parsed = (protocol.parse_chat_request if chat else protocol.parse_completion_request)(body, self.limits)
        model = self.models.get(parsed.model)

        request.info["model"] = model.name
        request.info["stream"] = parsed.stream

        try:
            self.limiter.acquire(principal.id)
        except APIError:
            self.metrics.rate_limited.inc(reason="principal")
            raise

        try:
            self.models.acquire_global()
        except APIError:
            self.limiter.release(principal.id)
            self.metrics.rate_limited.inc(reason="global")
            raise

        try:
            model.enter()
        except APIError:
            self.models.release_global()
            self.limiter.release(principal.id)
            raise

        released = False

        def release():
            nonlocal released

            if not released:
                released = True
                model.exit()
                self.models.release_global()
                self.limiter.release(principal.id)
                self.metrics.active.dec(model=model.name)

        self.metrics.active.inc(model=model.name)

        try:
            loop = asyncio.get_running_loop()
            hint = parsed.sampling.max_tokens or model.generator.default_config.max_new_tokens

            if chat:
                ids, dropped, tok_s = await loop.run_in_executor(
                    self._tok_pool, self._tokenize_chat, model, parsed.messages, hint
                )
            else:
                ids, dropped, tok_s = await loop.run_in_executor(
                    self._tok_pool, self._tokenize_completion, model, parsed.prompt
                )

            self.metrics.tokenize_s.observe(tok_s, model=model.name)
            self.metrics.prompt_len.observe(len(ids), model=model.name)
            request.info["messages_dropped"] = dropped

            gen_cfg = self._generation_config(model, parsed.sampling, len(ids), chat)
            timeout = self.limits.request_timeout_s

            gen_request = GenerationRequest(prompt=ids, config=gen_cfg, request_id=request.request_id,
                                            timeout_s=timeout, truncate_prompt=False)
            handle = model.generator.submit(gen_request)
            self.metrics.requests.inc(model=model.name, endpoint="chat" if chat else "completions",
                                      stream=str(parsed.stream).lower())

            runner = _Run(self, model, handle, request, ctx, chat, parsed, len(ids), release)

            if parsed.stream:
                return StreamingResponse(runner.stream(), on_close=runner.close)

            return await runner.collect()
        except BaseException:
            release()
            raise

    def on_request_done(self, request, status, duration, write_s, disconnected, streamed):
        route = request.info.get("route", "other")
        self.metrics.http_requests.inc(route=route, status=status)
        self.metrics.http_duration.observe(duration, route=route)

        if write_s:
            self.metrics.network_s.observe(write_s, route=route)

        if self._access is None:
            return

        record = {
            "ts": round(time.time(), 3),
            "request_id": request.request_id,
            "method": request.method,
            "route": route,
            "status": status,
            "duration_s": round(duration, 4),
            "network_write_s": round(write_s, 4),
            "client_disconnected": disconnected,
            "streamed": streamed,
            "client_ip": self._client_ip(request),
        }

        record.update({k: v for k, v in request.info.items() if k != "route"})

        try:
            self._access.write(_json_dumps(record) + "\n")
        except (OSError, ValueError):
            pass


class _Run:

    def __init__(self, app, model, handle, request, ctx, chat, parsed, prompt_len, release):
        self.app = app
        self.model = model
        self.handle = handle
        self.request = request
        self.ctx = ctx
        self.chat = chat
        self.parsed = parsed
        self.prompt_len = prompt_len
        self.release = release
        self.lstrip = _LStrip(chat and model.template.output_lstrip)
        self.rid = protocol.new_id("chatcmpl" if chat else "cmpl")
        self.created = int(time.time())
        self._abandoned = False
        self._done = False

    def close(self):
        self._abandon("client_disconnect")
        self.release()

    def _observe(self, ev):
        self._done = True
        m = self.app.metrics
        name = self.model.name
        h = self.handle
        now = time.perf_counter()
        usage = ev.usage or {}

        if h.admitted_at is not None:
            m.queue_s.observe(h.admitted_at - h.submitted_at, model=name)

        m.prefill_s.observe(h.prefill_s, model=name)

        ttft = usage.get("time_to_first_token_s")
        m.ttft_s.observe(ttft, model=name)
        m.generation_s.observe(now - h.submitted_at, model=name)

        completion = usage.get("completion_tokens", 0)
        m.prompt_tokens.inc(usage.get("prompt_tokens", 0) or 0, model=name)
        m.completion_tokens.inc(completion, model=name)

        if ttft is not None and completion > 1:
            span = (now - h.submitted_at) - ttft

            if span > 0:
                m.decode_rate.observe((completion - 1) / span, model=name)

        m.finished.inc(model=name, reason=ev.finish_reason or "unknown")

        if ev.finish_reason in ("cancelled", "timeout"):
            m.cancellations.inc(model=name, cause=ev.finish_reason)

        self.request.info.update({
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion,
            "finish_reason": ev.finish_reason,
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "queue_s": round(h.admitted_at - h.submitted_at, 4) if h.admitted_at else None,
        })

    def _usage(self, ev):
        u = ev.usage or {}
        return protocol.usage_dict(u.get("prompt_tokens", self.prompt_len), u.get("completion_tokens", 0))

    def _terminal_error(self, ev):
        if ev.finish_reason == "timeout":
            return APIError(504, "generation timed out", "server_error", code="timeout")

        if ev.finish_reason == "cancelled":
            return errors.overloaded("generation was cancelled because the model is unloading")

        log.error("generation %s failed: %s", self.request.request_id, ev.error)
        self.app.metrics.errors.inc(kind="generation")
        return errors.internal(self.request.request_id)

    def _abandon(self, cause):
        if not self._abandoned and not self._done and not self.handle.finished.is_set():
            self._abandoned = True
            self.handle.request_cancel()
            self.app.metrics.cancellations.inc(model=self.model.name, cause=cause)
            self.app.metrics.finished.inc(model=self.model.name, reason="client_disconnect")
            self.request.info["finish_reason"] = "client_disconnect"

    async def collect(self):
        bridge = _Bridge(self.handle, asyncio.get_running_loop())
        parts = []
        last = None

        try:
            async for ev in bridge.events(self.ctx, self.app.limits.sse_keepalive_s):
                if ev is None:
                    continue

                if ev.text:
                    parts.append(self.lstrip(ev.text))

                last = ev
        except ClientDisconnected:
            self._abandon("client_disconnect")
            raise
        finally:
            self.release()

        self._observe(last)

        if last.finish_reason not in FINISH_TO_OPENAI:
            raise self._terminal_error(last)

        text = "".join(parts)
        finish = FINISH_TO_OPENAI[last.finish_reason]
        usage = self._usage(last)
        fp = self.model.fingerprint

        if self.chat:
            body = protocol.chat_response(self.rid, self.model.name, fp, text, finish, usage)
        else:
            body = protocol.completion_response(self.rid, self.model.name, fp, text, finish, usage)

        return Response(200, body)

    def _chunk(self, text=None, finish=None, usage=None, include_choice=True, role=False):
        fp = self.model.fingerprint

        if self.chat:
            delta = {}

            if role:
                delta["role"] = "assistant"
                delta["content"] = ""

            if text:
                delta["content"] = text

            return protocol.chat_chunk(self.rid, self.created, self.model.name, fp, delta, finish, usage,
                                       include_choice)

        return protocol.completion_chunk(self.rid, self.created, self.model.name, fp, text or "", finish, usage,
                                         include_choice)

    async def stream(self):
        bridge = _Bridge(self.handle, asyncio.get_running_loop())
        finished = False

        try:
            if self.chat:
                yield sse(self._chunk(role=True))

            async for ev in bridge.events(self.ctx, self.app.limits.sse_keepalive_s):
                if ev is None:
                    yield ": keep-alive\n\n"
                    continue

                text = self.lstrip(ev.text) if ev.text else ""

                if not ev.finished:
                    if text:
                        yield sse(self._chunk(text=text))
                    continue

                finished = True
                self._observe(ev)

                if ev.finish_reason in FINISH_TO_OPENAI:
                    if text:
                        yield sse(self._chunk(text=text))

                    yield sse(self._chunk(finish=FINISH_TO_OPENAI[ev.finish_reason]))

                    if self.parsed.include_usage:
                        yield sse(self._chunk(usage=self._usage(ev), include_choice=False))
                else:
                    if text:
                        yield sse(self._chunk(text=text))

                    err = self._terminal_error(ev)
                    self.request.info["error_code"] = err.code
                    yield sse(err.to_dict())

                yield "data: [DONE]\n\n"
                return
        except ClientDisconnected:
            pass
        finally:
            if not finished:
                self._abandon("client_disconnect")

            self.release()
