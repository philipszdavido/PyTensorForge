import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from pytensorforge.inference.config import GenerationConfig
from pytensorforge.inference.kv_cache import KVCacheManager
from pytensorforge.inference.sampling import sample_token
from pytensorforge.inference.text import StreamDecoder


@dataclass
class GenerationRequest:
    prompt: Union[str, List[int]]
    config: GenerationConfig = field(default_factory=GenerationConfig)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timeout_s: Optional[float] = None
    truncate_prompt: bool = False


@dataclass
class StreamEvent:
    request_id: str
    token_id: Optional[int]
    text: str
    finished: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class GenerationResult:
    request_id: str
    text: str
    token_ids: List[int]
    finish_reason: str
    usage: dict
    error: Optional[str] = None


class RequestHandle:

    def __init__(self, request, scheduler):
        self.request = request
        self.request_id = request.request_id
        self._scheduler = scheduler
        self._events = queue.Queue()
        self.finished = threading.Event()
        self.submitted_at = time.perf_counter()
        self.admitted_at = None
        self.prefill_s = None
        self.prompt_tokens = None
        self._listener = None

    def cancel(self):
        self._scheduler.cancel(self.request_id)

    def request_cancel(self):
        self._scheduler.request_cancel(self.request_id)

    def set_listener(self, fn):
        self._listener = fn

        if fn is not None and (not self._events.empty() or self.finished.is_set()):
            fn()

    def _emit(self, ev):
        self._events.put(ev)
        fn = self._listener

        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def drain(self):
        out = []

        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def events(self, drive=True, poll_s=0.02):
        try:
            while True:
                try:
                    ev = self._events.get_nowait()
                except queue.Empty:
                    if self.finished.is_set():
                        return

                    if drive:
                        progressed = self._scheduler.step()

                        if not progressed and self._events.empty():
                            self.finished.wait(poll_s)

                        continue

                    try:
                        ev = self._events.get(timeout=poll_s)
                    except queue.Empty:
                        continue

                yield ev

                if ev.finished:
                    return
        finally:
            if not self.finished.is_set():
                self.cancel()

    def result(self, drive=True):
        parts = []
        ids = []
        last = None

        for ev in self.events(drive=drive):
            parts.append(ev.text)
            if ev.token_id is not None:
                ids.append(ev.token_id)
            last = ev

        return GenerationResult(
            request_id=self.request_id,
            text="".join(parts),
            token_ids=ids,
            finish_reason=last.finish_reason,
            usage=last.usage,
            error=last.error,
        )


class _Sequence:

    def __init__(self, handle, prompt_ids, cache, decoder, rng, deadline):
        self.handle = handle
        self.request = handle.request
        self.config = handle.request.config
        self.prompt_ids = prompt_ids
        self.cache = cache
        self.decoder = decoder
        self.rng = rng
        self.deadline = deadline
        self.generated = []
        self.seen = set(prompt_ids)
        self.last_token = None
        self.submitted = time.perf_counter()
        self.first_token_at = None


class BatchScheduler:

    def __init__(
        self,
        engine,
        tokenizer,
        max_batch_size=8,
        cache_budget_bytes=None,
        prefill_chunk=256,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.prefill_chunk = prefill_chunk

        self.cache_manager = KVCacheManager(
            engine.n_layers, engine.n_heads, engine.head_dim, max_bytes=cache_budget_bytes
        )

        self._waiting = deque()
        self._waiting_lock = threading.Lock()
        self._active = {}
        self._step_lock = threading.RLock()
        self._runner = None
        self._stop = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_requests = set()

        self.stats = {
            "requests_submitted": 0,
            "requests_completed": 0,
            "requests_cancelled": 0,
            "requests_errored": 0,
            "requests_timed_out": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "prefill_seconds": 0.0,
            "decode_seconds": 0.0,
            "decode_steps": 0,
            "max_batch_seen": 0,
            "ttft_seconds_total": 0.0,
            "ttft_count": 0,
        }

    def submit(self, request):
        handle = RequestHandle(request, self)

        with self._waiting_lock:
            self._waiting.append(handle)
            self.stats["requests_submitted"] += 1

        return handle

    @property
    def num_active(self):
        return len(self._active)

    @property
    def num_waiting(self):
        return len(self._waiting)

    def _usage(self, seq_or_none, handle, prompt_len=0, completion=0, ttft=None):
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion,
            "total_tokens": prompt_len + completion,
            "time_to_first_token_s": ttft,
        }

    def _finalize_unstarted(self, handle, reason, error=None):
        handle._emit(StreamEvent(
            handle.request_id, None, "", True, reason, self._usage(None, handle), error
        ))
        handle.finished.set()

        key = {"cancelled": "requests_cancelled", "timeout": "requests_timed_out"}.get(reason)
        self.stats[key or "requests_errored"] += 1

    def _finish(self, seq, reason, error=None, token_id=None, text=""):
        tail = seq.decoder.flush()
        ttft = None if seq.first_token_at is None else seq.first_token_at - seq.submitted

        seq.handle._emit(StreamEvent(
            seq.request.request_id,
            token_id,
            text + tail,
            True,
            reason,
            self._usage(seq, seq.handle, len(seq.prompt_ids), len(seq.generated), ttft),
            error,
        ))
        seq.handle.finished.set()

        self.cache_manager.release(seq.request.request_id)
        self._active.pop(seq.request.request_id, None)

        key = {"cancelled": "requests_cancelled", "timeout": "requests_timed_out", "error": "requests_errored"}
        self.stats[key.get(reason, "requests_completed")] += 1

    def cancel(self, request_id):
        with self._step_lock:
            seq = self._active.get(request_id)

            if seq is not None:
                self._finish(seq, "cancelled")
                return

            with self._waiting_lock:
                for handle in list(self._waiting):
                    if handle.request_id == request_id:
                        self._waiting.remove(handle)
                        self._finalize_unstarted(handle, "cancelled")
                        return

    def request_cancel(self, request_id):
        with self._cancel_lock:
            self._cancel_requests.add(request_id)

    def _apply_cancel_requests(self):
        with self._cancel_lock:
            if not self._cancel_requests:
                return
            pending = self._cancel_requests
            self._cancel_requests = set()

        for request_id in pending:
            self.cancel(request_id)

    def _prepare_prompt(self, request):
        ctx = self.engine.context_length
        cfg = request.config

        if isinstance(request.prompt, str):
            ids = self.tokenizer.encode(request.prompt)
        else:
            ids = [int(t) for t in request.prompt]

        if not ids:
            if self.tokenizer.eos_id is None:
                raise ValueError("empty prompt and tokenizer has no EOS token to start from")
            ids = [self.tokenizer.eos_id]

        if max(ids) >= self.engine.config.vocab_size or min(ids) < 0:
            raise ValueError("prompt contains token ids outside the model vocabulary")

        if len(ids) >= ctx:
            if not request.truncate_prompt:
                raise ValueError(f"prompt of {len(ids)} tokens does not fit the context length of {ctx}")

            keep = ctx - min(cfg.max_new_tokens, ctx // 2)
            ids = ids[-keep:]

        return ids

    def _admit(self):
        while len(self._active) < self.max_batch_size:
            with self._waiting_lock:
                if not self._waiting:
                    return
                handle = self._waiting[0]

            req = handle.request

            try:
                ids = self._prepare_prompt(req)
            except Exception as exc:
                with self._waiting_lock:
                    self._waiting.popleft()
                self._finalize_unstarted(handle, "error", str(exc))
                continue

            capacity = min(len(ids) + req.config.max_new_tokens, self.engine.context_length)

            if not self.cache_manager.fits_at_all(capacity):
                with self._waiting_lock:
                    self._waiting.popleft()
                self._finalize_unstarted(handle, "error", "request needs more kv cache than the configured budget")
                continue

            if not self.cache_manager.can_allocate(capacity):
                return

            with self._waiting_lock:
                self._waiting.popleft()

            cache = self.cache_manager.allocate(req.request_id, capacity)
            handle.admitted_at = time.perf_counter()
            handle.prompt_tokens = len(ids)
            deadline = handle.submitted_at + req.timeout_s if req.timeout_s else None
            seq = _Sequence(
                handle,
                ids,
                cache,
                StreamDecoder(self.tokenizer, req.config.stop_strings),
                np.random.default_rng(req.config.seed),
                deadline,
            )
            self._active[req.request_id] = seq
            self.stats["prompt_tokens"] += len(ids)

            started = time.perf_counter()

            try:
                logits = self.engine.prefill(ids, cache, self.prefill_chunk)
            except Exception as exc:
                self._finish(seq, "error", str(exc))
                continue

            handle.prefill_s = time.perf_counter() - started
            self.stats["prefill_seconds"] += handle.prefill_s
            self._consume(seq, logits)

    def _consume(self, seq, logits):
        cfg = seq.config
        token = sample_token(logits, cfg, seq.rng, seq.seen)

        if seq.first_token_at is None:
            seq.first_token_at = time.perf_counter()
            self.stats["ttft_seconds_total"] += seq.first_token_at - seq.submitted
            self.stats["ttft_count"] += 1

        seq.generated.append(token)
        seq.seen.add(token)
        seq.last_token = token
        self.stats["generated_tokens"] += 1

        is_stop_token = token in cfg.stop_token_ids or (cfg.stop_on_eos and token == self.tokenizer.eos_id)

        delta = "" if is_stop_token else seq.decoder.push(token)

        if is_stop_token or seq.decoder.stopped:
            self._finish(seq, "stop", token_id=token, text=delta)
            return

        if len(seq.generated) >= cfg.max_new_tokens or seq.cache.length + 1 >= seq.cache.capacity:
            self._finish(seq, "length", token_id=token, text=delta)
            return

        seq.handle._emit(StreamEvent(seq.request.request_id, token, delta))

    def step(self):
        with self._step_lock:
            self._apply_cancel_requests()
            now = time.perf_counter()

            with self._waiting_lock:
                expired = [
                    h for h in self._waiting
                    if h.request.timeout_s and now - h.submitted_at > h.request.timeout_s
                ]
                for h in expired:
                    self._waiting.remove(h)

            for h in expired:
                self._finalize_unstarted(h, "timeout")

            for seq in list(self._active.values()):
                if seq.deadline is not None and now > seq.deadline:
                    self._finish(seq, "timeout")

            self._admit()

            seqs = list(self._active.values())

            if not seqs:
                return bool(self._waiting)

            self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(seqs))
            started = time.perf_counter()

            try:
                logits = self.engine.decode_batch([s.last_token for s in seqs], [s.cache for s in seqs])
            except Exception as exc:
                for s in seqs:
                    self._finish(s, "error", str(exc))
                return True

            self.stats["decode_seconds"] += time.perf_counter() - started
            self.stats["decode_steps"] += 1

            for s, row in zip(seqs, logits):
                self._consume(s, row)

            return True

    def run_until_idle(self):
        while self.step():
            pass

    def start(self, idle_sleep_s=0.002):
        if self._runner is not None:
            return

        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                if not self.step():
                    time.sleep(idle_sleep_s)

        self._runner = threading.Thread(target=loop, daemon=True)
        self._runner.start()

    def stop(self):
        self._stop.set()

        if self._runner is not None:
            self._runner.join(timeout=2.0)
            self._runner = None

    @property
    def background(self):
        return self._runner is not None
