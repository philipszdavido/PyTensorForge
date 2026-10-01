import bisect
import math
import os
import resource
import threading
import time

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
RATE_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
TOKEN_BUCKETS = (16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _escape(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names, values):
    if not names:
        return ""

    return "{" + ",".join(f'{n}="{_escape(v)}"' for n, v in zip(names, values)) + "}"


def _fmt(v):
    if v == math.inf:
        return "+Inf"

    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))

    return repr(float(v)) if isinstance(v, float) else str(v)


class _Metric:
    kind = ""

    def __init__(self, name, help_text, labels=()):
        self.name = name
        self.help = help_text
        self.label_names = tuple(labels)
        self._lock = threading.Lock()

    def _key(self, labels):
        if set(labels) != set(self.label_names):
            raise ValueError(f"{self.name} expects labels {self.label_names}, got {tuple(labels)}")

        return tuple(str(labels[n]) for n in self.label_names)


class Counter(_Metric):
    kind = "counter"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._values = {}

    def inc(self, amount=1, **labels):
        key = self._key(labels)

        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def get(self, **labels):
        return self._values.get(self._key(labels), 0)

    def samples(self):
        with self._lock:
            return [(self.name + "_total", k, v) for k, v in sorted(self._values.items())]

    def snapshot(self):
        with self._lock:
            return {",".join(k) or "": v for k, v in self._values.items()}


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, *a, fn=None, **k):
        super().__init__(*a, **k)
        self._values = {}
        self._fn = fn

    def set(self, value, **labels):
        key = self._key(labels)

        with self._lock:
            self._values[key] = value

    def inc(self, amount=1, **labels):
        key = self._key(labels)

        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount=1, **labels):
        self.inc(-amount, **labels)

    def remove(self, **labels):
        with self._lock:
            self._values.pop(self._key(labels), None)

    def _current(self):
        if self._fn is not None:
            return {self._key(lbl): v for lbl, v in self._fn()}

        with self._lock:
            return dict(self._values)

    def samples(self):
        return [(self.name, k, v) for k, v in sorted(self._current().items())]

    def snapshot(self):
        return {",".join(k) or "": v for k, v in self._current().items()}


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, *a, buckets=LATENCY_BUCKETS, **k):
        super().__init__(*a, **k)
        self.buckets = tuple(sorted(buckets))
        self._data = {}

    def observe(self, value, **labels):
        if value is None or not math.isfinite(value):
            return

        key = self._key(labels)
        idx = bisect.bisect_left(self.buckets, value)

        with self._lock:
            d = self._data.get(key)

            if d is None:
                d = self._data[key] = [[0] * (len(self.buckets) + 1), 0.0, 0]

            d[0][idx] += 1
            d[1] += value
            d[2] += 1

    def samples(self):
        out = []

        with self._lock:
            items = sorted((k, (list(v[0]), v[1], v[2])) for k, v in self._data.items())

        for key, (counts, total, n) in items:
            running = 0

            for bound, c in zip(self.buckets + (math.inf,), counts):
                running += c
                out.append((self.name + "_bucket", key + (_fmt(bound),), running))

            out.append((self.name + "_sum", key, total))
            out.append((self.name + "_count", key, n))

        return out

    def label_names_for(self, sample_name):
        if sample_name.endswith("_bucket"):
            return self.label_names + ("le",)

        return self.label_names

    def snapshot(self):
        with self._lock:
            return {
                ",".join(k) or "": {"count": v[2], "sum": v[1], "mean": (v[1] / v[2]) if v[2] else None}
                for k, v in self._data.items()
            }


class Registry:

    def __init__(self, namespace="ptf"):
        self.namespace = namespace
        self._metrics = []

    def _add(self, metric):
        metric.name = f"{self.namespace}_{metric.name}"
        self._metrics.append(metric)
        return metric

    def counter(self, name, help_text, labels=()):
        return self._add(Counter(name, help_text, labels))

    def gauge(self, name, help_text, labels=(), fn=None):
        return self._add(Gauge(name, help_text, labels, fn=fn))

    def histogram(self, name, help_text, labels=(), buckets=LATENCY_BUCKETS):
        return self._add(Histogram(name, help_text, labels, buckets=buckets))

    def render_prometheus(self):
        lines = []

        for m in self._metrics:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} {m.kind}")

            for sample_name, key, value in m.samples():
                names = m.label_names_for(sample_name) if isinstance(m, Histogram) else m.label_names
                lines.append(f"{sample_name}{_labels(names, key)} {_fmt(value)}")

        return "\n".join(lines) + "\n"

    def snapshot(self):
        return {m.name: m.snapshot() for m in self._metrics}


def _rss_bytes():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class ServerMetrics:

    def __init__(self, model_server_ref):
        r = self.registry = Registry("ptf")
        self._server = model_server_ref
        self.started = time.time()

        self.http_requests = r.counter("http_requests", "HTTP requests by route and status", ("route", "status"))
        self.http_duration = r.histogram("http_request_duration_seconds", "Full HTTP request duration", ("route",))
        self.open_connections = r.gauge("open_connections", "Open client TCP connections")

        self.requests = r.counter("completion_requests", "Completion requests accepted", ("model", "endpoint", "stream"))
        self.finished = r.counter("completion_finished", "Completions by finish reason", ("model", "reason"))
        self.cancellations = r.counter("cancellations", "Generations cancelled before finishing", ("model", "cause"))
        self.errors = r.counter("errors", "Errors by kind", ("kind",))
        self.auth_failures = r.counter("auth_failures", "Rejected authentication attempts")
        self.rate_limited = r.counter("rate_limited", "Requests rejected by rate limits", ("reason",))

        self.prompt_tokens = r.counter("prompt_tokens", "Prompt tokens processed", ("model",))
        self.completion_tokens = r.counter("completion_tokens", "Completion tokens generated", ("model",))
        self.active = r.gauge("active_requests", "Requests in flight (queued or generating)", ("model",))

        self.tokenize_s = r.histogram("tokenize_seconds", "Chat templating + prompt tokenization time", ("model",))
        self.queue_s = r.histogram("queue_wait_seconds", "Time from submit to admission into a batch", ("model",))
        self.prefill_s = r.histogram("prefill_seconds", "Prompt prefill compute time", ("model",))
        self.ttft_s = r.histogram("time_to_first_token_seconds", "Submit to first generated token", ("model",))
        self.generation_s = r.histogram("generation_duration_seconds", "Submit to final token", ("model",))
        self.decode_rate = r.histogram("decode_tokens_per_second", "Per-request generation rate after first token",
                                       ("model",), buckets=RATE_BUCKETS)
        self.network_s = r.histogram("network_write_seconds", "Time blocked writing response bytes to the client",
                                     ("route",))
        self.prompt_len = r.histogram("prompt_length_tokens", "Prompt length", ("model",), buckets=TOKEN_BUCKETS)

        r.gauge("process_resident_memory_bytes", "Resident set size", fn=lambda: [({}, _rss_bytes())])
        r.gauge("process_cpu_seconds", "CPU seconds used by the server process",
                fn=lambda: [({}, time.process_time())])
        r.gauge("uptime_seconds", "Seconds since server start", fn=lambda: [({}, time.time() - self.started)])

        self._scheduler_gauges(r)

    def _each_model(self):
        server = self._server()

        if server is None:
            return []

        return server.loaded_models()

    def _scheduler_gauges(self, r):
        def per_model(fn):
            return lambda: [({"model": m.name}, fn(m)) for m in self._each_model()]

        def sched(key):
            return per_model(lambda m: m.generator.scheduler.stats[key])

        r.gauge("scheduler_waiting", "Requests waiting for a batch slot or cache", ("model",),
                fn=per_model(lambda m: m.generator.scheduler.num_waiting))
        r.gauge("scheduler_running", "Sequences in the running batch", ("model",),
                fn=per_model(lambda m: m.generator.scheduler.num_active))
        r.gauge("kv_cache_used_bytes", "KV cache bytes allocated", ("model",),
                fn=per_model(lambda m: m.generator.scheduler.cache_manager.used_bytes))
        r.gauge("kv_cache_budget_bytes", "KV cache byte budget", ("model",),
                fn=per_model(lambda m: m.generator.scheduler.cache_manager.max_bytes or 0))
        r.gauge("model_weights_bytes", "Weights resident in memory", ("model",),
                fn=per_model(lambda m: m.generator.weights_nbytes))
        r.gauge("scheduler_decode_seconds", "Cumulative batched decode compute", ("model",),
                fn=sched("decode_seconds"))
        r.gauge("scheduler_prefill_seconds", "Cumulative prefill compute", ("model",),
                fn=sched("prefill_seconds"))
        r.gauge("scheduler_decode_steps", "Batched decode steps executed", ("model",),
                fn=sched("decode_steps"))
        r.gauge("scheduler_mean_batch_size", "Generated tokens per decode step", ("model",),
                fn=per_model(lambda m: (m.generator.scheduler.stats["generated_tokens"]
                                        / max(1, m.generator.scheduler.stats["decode_steps"]))))
        r.gauge("scheduler_max_batch_seen", "Largest decode batch so far", ("model",),
                fn=sched("max_batch_seen"))
