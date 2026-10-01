# Phase 4 — Model Server, OpenAI-Compatible API, Web Client

Phase 4 puts a production-oriented serving stack on top of the Phase 3 inference
runtime. It covers spec sections 35–43: the HTTP API, OpenAI compatibility, the
chat web client, SSE streaming with cancellation, chat templates, the layered
model server, resource management, security, and observability.

No new runtime dependencies were added. The HTTP server is built on the Python
standard library (`asyncio`); the project still depends only on `numpy` and
`pyyaml`. The web client is TypeScript, and the compiled JavaScript is committed
in `web/dist`, so running the UI needs no Node toolchain.

## Layering

```
src/models/gpt          Transformer (no I/O, no HTTP)
src/training            training runtime
src/inference           InferenceModel, KV cache, BatchScheduler, sampling,
                        chat_template.py, export/runtime (load_model)
src/serving
  model_server.py       ModelServer: load/unload, limits, device, memory budget
  protocol.py           OpenAI request parsing + response/chunk builders (no sockets)
  security.py           KeyStore, RateLimiter
  metrics.py            Prometheus/JSON metrics registry
  app.py                routing, auth, CORS, SSE, cancellation bridge
  http.py               asyncio HTTP/1.1 transport
  server.py             process lifecycle, signals, graceful drain
web/                    TypeScript chat client (talks HTTP/SSE only)
```

Each layer only calls downward. The Transformer knows nothing about chat
templates; the scheduler knows nothing about HTTP; `protocol.py` can be driven
from a test without a socket. The same exported model directory is used by
Python (`load_model`), the CLI (`generate`), the HTTP API, and the web client.

## Quick start

```bash
pytensorforge export checkpoints/latest --output exports/my-gpt --tokenizer tokenizer/ --chat-template ptf-chat

echo "$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" > api_keys.txt
pytensorforge serve exports/my-gpt --name my-gpt --api-key-file api_keys.txt
```

Open `http://127.0.0.1:8000/` for the chat UI, or use any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=open("api_keys.txt").read().strip())
for chunk in client.chat.completions.create(
        model="my-gpt", messages=[{"role": "user", "content": "Hello"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

For multi-model or production deployments use a config file:
`pytensorforge serve --config serve.example.yaml`. CLI flags override the
matching config values. Unknown config keys are rejected rather than ignored.

`serve` binds `127.0.0.1` by default. Binding a non-loopback address with no
API keys configured is refused at startup unless `--insecure-no-auth`
(`security.allow_unauthenticated: true`) is given explicitly.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness (process is up) |
| GET | `/ready` | none | readiness (≥1 model loaded, not draining) |
| GET | `/metrics` | API key (configurable) | Prometheus text; `?format=json` for JSON |
| GET | `/v1/models`, `/v1/models/{id}` | API key | loaded models with context/generation limits |
| POST | `/v1/chat/completions` | API key | chat, blocking or SSE |
| POST | `/v1/completions` | API key | raw prompt completion, blocking or SSE |
| GET | `/admin/models` | admin key | all declared models with load state and memory |
| POST | `/admin/models/{name}/load` | admin key | load a model declared in the config |
| POST | `/admin/models/{name}/unload` | admin key | drain, cancel stragglers, free memory |
| GET | `/`, static assets | none | web client (indexed allowlist, strict CSP) |

The admin API can only load models declared in the server config. It never
accepts a filesystem path over HTTP.

### OpenAI compatibility

Supported request fields: `model`, `messages` (roles `system`, `developer`
(treated as system), `user`, `assistant`; string content or text content
parts), `prompt` (completions), `temperature`, `top_p`, `max_tokens`,
`max_completion_tokens`, `stream`, `stream_options.include_usage`, `stop`
(string or up to 4 strings), `seed`, `n` (only 1), and the PyTensorForge
extensions `top_k` and `repetition_penalty`.

Parameters that the runtime cannot honour are rejected with a
`400 invalid_request_error` naming the parameter, rather than silently ignored:
`tools`, `tool_choice`, `functions`, `function_call`, `audio`, `modalities`,
`prediction`, and any of `n`, `presence_penalty`, `frequency_penalty`,
`logprobs`, `top_logprobs`, `logit_bias`, `echo`, `best_of`, `suffix` set to a
non-default value. Metadata-only fields (`user`, `metadata`, `store`,
`service_tier`, `response_format`) are accepted and ignored.

Responses have the OpenAI shape: `id`, `object`, `created`, `model`, `choices`
with `message`/`delta`/`text`, `finish_reason` (`stop`, `length`), and `usage`
(`prompt_tokens`, `completion_tokens`, `total_tokens`). Streaming responses are
`text/event-stream` with `data: {chunk}` events terminated by `data: [DONE]`;
with `include_usage` a final chunk carries usage and an empty `choices`.
Errors use `{"error": {"message", "type", "param", "code"}}`.

Compatibility was verified against the official `openai` Python SDK (blocking,
streaming, usage chunk, error mapping) in `test/gpt/test_phase4_server.py`.

## Chat templates

Chat formatting lives in `src/inference/chat_template.py`, not in the engine.
A template is declarative JSON: role prefixes/suffixes built from literal text
and `{"special": "<|token|>"}` references, a generation prompt, stop tokens,
stop strings, and flags. No code or Jinja is executed.

```json
{
  "name": "ptf-chat",
  "add_bos": false,
  "roles": {
    "system":    {"prefix": [{"special": "<|system|>"}, "\n"],    "suffix": [{"special": "<|end|>"}, "\n"]},
    "user":      {"prefix": [{"special": "<|user|>"}, "\n"],      "suffix": [{"special": "<|end|>"}, "\n"]},
    "assistant": {"prefix": [{"special": "<|assistant|>"}, "\n"], "suffix": [{"special": "<|end|>"}, "\n"]}
  },
  "generation_prompt": [{"special": "<|assistant|>"}, "\n"],
  "stop_tokens": ["<|end|>"],
  "stop_strings": [],
  "output_lstrip": false,
  "default_system": null
}
```

Two built-ins ship: `plain` (`User:` / `Assistant:` text, works with any
tokenizer, stops on `\nUser:`) and `ptf-chat` (special tokens; requires a
tokenizer that defines them, otherwise binding raises `ChatTemplateError`).
The template is chosen, in order, by the model entry's `chat_template`, the
template stored in the export's `config.json` (`export --chat-template`), and
finally `runtime.fallback_chat_template`. A value may be a built-in name, an
inline object, or a path to a JSON file in the config.

**Injection safety.** Message content is always encoded as ordinary text.
Special-token ids are only produced from the template's structural
`{"special": ...}` parts, and any control ids that appear inside encoded user
content are filtered, so a user typing `<|assistant|>` cannot forge a turn.

**Truncation.** With `chat_truncation: auto`, when a conversation exceeds the
prompt budget the renderer keeps all system messages and the newest turns,
dropping the oldest non-system messages. If even the pinned messages plus the
last message do not fit, or truncation is `disabled`, the request fails with
`400 context_length_exceeded`.

## Streaming and cancellation

The HTTP layer never blocks on the model. Tokenization runs in a thread pool;
the scheduler runs its batched decode loop in its own thread; a small bridge
moves `StreamEvent`s onto the event loop.

Each connection has a disconnect watcher. When the client closes the socket
during a streaming or a blocking request, the server calls the scheduler's
thread-safe `request_cancel(id)`. The cancel is applied at the start of the next
scheduler step, the sequence leaves the batch, and its KV cache blocks are
released immediately. Tests assert that after a disconnect the scheduler has no
active sequences and `kv_cache_used_bytes` returns to zero.

SSE keep-alive comments (`: keep-alive`) are sent every `sse_keepalive_s` while a
request is queued or prefilling, so proxies do not time out long prompts.
If `request_timeout_s` is exceeded, a blocking request returns `504`; a
streaming request that has already sent headers receives a final
`data: {"error": ...}` event and `[DONE]`, and generation is cancelled.
Server errors never expose tracebacks, exception text, or paths — clients get a
generic message with a request id that matches the server log line.

## Resource management

- **Model load/unload.** `ModelServer` loads declared models at startup
  (`preload`) or via the admin API. Unload stops admitting new requests, waits
  up to `shutdown_drain_s` for in-flight requests, cancels the rest, then drops
  weights and the KV cache.
- **Memory limits.** `runtime.memory_limit_mb` bounds weights + KV-cache budgets
  across all loaded models. A load that would exceed it fails with
  `507` (code `insufficient_memory`) before allocating. Each model's KV cache has a
  hard byte budget (`kv_cache_budget_mb`, default: enough for a full batch at
  full context); requests wait for cache rather than over-allocating.
- **Concurrency.** `limits.max_concurrent_requests` (global) and
  `max_batch_size` (per model) are hard caps. Excess requests get
  `503 overloaded` with `Retry-After`.
- **Context and generation limits.** Per-model `max_context_tokens` and
  `max_generation_tokens`, capped by the model's trained context and the global
  `limits.max_generation_tokens`. `max_tokens` above the limit is rejected, not
  clamped; omitting it uses the remaining context up to the limit.
- **Request size.** Body size (`413`), header size (`431`), URL length (`414`),
  message count, prompt characters, and prompt tokens are all bounded before any
  model work. Header and body read timeouts defeat slow-loris clients (`408`).
- **Device.** `runtime.device` is `auto` or `cpu`. Requesting an accelerator
  fails at startup with a clear message, because the backend is NumPy/CPU.

## Security

- **API keys.** Loaded from `security.api_keys`, `api_keys_file` (one per line,
  `#` comments allowed), and the `PTF_API_KEYS` environment variable
  (comma-separated). Keys are stored only as SHA-256 digests; entries may be
  given pre-hashed as `sha256:<hex>` so plaintext never sits in the config.
  Minimum length 16. Comparison is constant-time. Clients send
  `Authorization: Bearer <key>` or `x-api-key: <key>`.
- **Admin keys** are a separate set and are required for `/admin/*`.
- **Rate limiting.** Per-principal token bucket (`requests_per_minute`,
  `burst`) plus per-principal concurrency, keyed by API key digest or client IP
  when unauthenticated. The tracking table is a bounded LRU, so a flood of
  distinct clients cannot grow memory. `X-Forwarded-For` is ignored unless
  `trust_forwarded_for` is set (only enable behind your own proxy).
- **CORS** is an explicit origin allowlist; no wildcard is sent with
  credentials.
- **Web UI** is served from a fixed file index (no path traversal) with
  `Content-Security-Policy: default-src 'none'; script-src 'self'; ...`,
  `X-Content-Type-Options: nosniff`, and `frame-ancestors 'none'`. The browser
  test confirms no CSP violations occur.
- **Logs** are JSONL access logs; prompt text is excluded unless
  `logging.log_prompts: true`.

For internet exposure, terminate TLS in a reverse proxy (nginx, Caddy) in front
of the server. The built-in server does not speak TLS.

## Observability

`/metrics` exports Prometheus text (namespace `ptf_`); `?format=json` returns
the same data as JSON. Histograms carry a `model` label.

| Metric | Meaning |
|---|---|
| `ptf_http_requests_total{route,status}`, `ptf_http_request_duration_seconds` | request counts and full latency |
| `ptf_open_connections` | open TCP connections |
| `ptf_completion_requests_total{model,endpoint,stream}` | accepted completion requests |
| `ptf_completion_finished_total{model,reason}` | finish reasons |
| `ptf_cancellations_total{model,cause}` | cancellations by cause (`client_disconnect`, `timeout`, `cancelled` for unload/shutdown) |
| `ptf_errors_total{kind}`, `ptf_auth_failures_total`, `ptf_rate_limited_total{reason}` | failures |
| `ptf_active_requests` | in-flight requests |
| `ptf_prompt_tokens_total`, `ptf_completion_tokens_total` | token counters |
| `ptf_tokenize_seconds` | chat templating + tokenization |
| `ptf_queue_wait_seconds` | submit → admitted into a batch |
| `ptf_prefill_seconds` | prompt prefill compute |
| `ptf_time_to_first_token_seconds` | submit → first token |
| `ptf_generation_duration_seconds` | submit → last token |
| `ptf_decode_tokens_per_second` | per-request decode rate after first token |
| `ptf_network_write_seconds` | time blocked writing to the client |
| `ptf_prompt_length_tokens` | prompt length distribution |
| `ptf_scheduler_waiting`, `ptf_scheduler_running`, `ptf_scheduler_mean_batch_size` | batching state |
| `ptf_scheduler_prefill_seconds`, `ptf_scheduler_decode_seconds` | cumulative model compute |
| `ptf_kv_cache_used_bytes`, `ptf_kv_cache_budget_bytes`, `ptf_model_weights_bytes` | memory |
| `ptf_process_resident_memory_bytes`, `ptf_process_cpu_seconds`, `ptf_uptime_seconds` | process |

### Attributing a bottleneck

TTFT decomposes as tokenize + queue wait + prefill (+ first decode step), so
compare those histograms first.

- **Tokenization**: `tokenize_seconds` is a large share of TTFT.
- **Batching / capacity**: `queue_wait_seconds` grows while
  `scheduler_waiting > 0` and `scheduler_running` sits at `max_batch_size`.
- **KV cache**: requests are waiting while `kv_cache_used_bytes` is at
  `kv_cache_budget_bytes` and the batch is not full — raise the budget or lower
  context limits.
- **Model computation / CPU / device**: `prefill_seconds` and
  `scheduler_decode_seconds` dominate wall time and `process_cpu_seconds` rises
  at about one core per second of compute. Low `decode_tokens_per_second` with a
  small mean batch size is compute-bound per sequence; a low rate with a large
  batch means batching is trading latency for throughput.
- **Network**: `network_write_seconds` is significant, or
  `http_request_duration_seconds` greatly exceeds `generation_duration_seconds`
  — slow clients or proxy buffering.
- **Batching efficiency**: `scheduler_mean_batch_size` near 1 under load means
  requests are not overlapping (check concurrency limits and arrival pattern).

## Web client

`web/` is a dependency-free browser app written in TypeScript:

- `sse.ts` — spec-conforming SSE parser (CR/LF/CRLF, multi-line data, comments).
- `api.ts` — `ChatClient`: model listing and streaming chat over `fetch`, typed
  `ApiError`/`NetworkError`/`AbortedError`, automatic retry with `Retry-After`
  for 429/502/503/504 and for network failures before any output arrives.
- `markdown.ts` — Markdown renderer that builds DOM nodes directly (never
  `innerHTML`): fenced code blocks with copy buttons, inline code, headings,
  nested lists, quotes, rules, and links restricted to http/https/mailto.
- `store.ts` — conversations and settings in `localStorage`; the API key is kept
  in `sessionStorage` unless the user opts to remember it.
- `app.ts` — UI: conversation list, new chat, streaming render, stop (aborts the
  fetch, which cancels server generation), regenerate, retry on failure,
  token/state display, connection status, settings (model, temperature, top-p,
  max tokens, system prompt).

The client only knows the OpenAI-style HTTP API, so it can be pointed at any
compatible server. To rebuild after editing: `cd web && npm install && npm run
build`; `npm test` runs the markdown XSS test suite (jsdom).

## Tests

```bash
python -m test.gpt.test_phase4_server
python -m test.gpt.test_phase4_browser
cd web && npm test
```

`test_phase4_server` covers templates (rendering, injection, truncation,
export round-trip), protocol validation, key store and rate limiter, Prometheus
format, memory-limit/device/insecure-bind refusals, full HTTP end-to-end (server
output equals direct runtime greedy output; streaming equals blocking; usage
chunk; seeded reproducibility; stop sequences; keep-alive; 411/413/100-continue;
CORS; UI + CSP; path traversal), disconnect cancellation freeing the KV cache,
503/429/504, non-leaking errors, admin load/unload, slow-loris 408, 431,
graceful drain, the official OpenAI SDK, and the Node.js client.

`test_phase4_browser` drives the real UI in headless Chromium (Playwright):
API-key prompt, streaming, regenerate, persistence across reload, stop
cancelling server-side generation, offline error with retry, and no console or
CSP errors. It skips cleanly if Playwright or Chromium is not installed.

The Phase 2, Phase 3, and tokenizer suites pass unchanged.

## Known limits

- **CPU only.** The engine is NumPy; there is no GPU, FP16/BF16, or activation
  checkpointing path yet, so `device` accepts only `auto`/`cpu`.
- **Attention in decode** (measured and kept deliberately; see `PHASE5_TRAINING_EFFICIENCY.md`) loops over sequences in the batch for the attention
  step (projections and MLP are batched). There is no paged attention, prefix
  caching, or preemption/swap of running sequences.
- **Request bodies** must have `Content-Length`; chunked *request* bodies get
  `411`. Chunked *responses* are fully supported.
- **Half-closed clients** (TCP FIN after sending the request) are treated as a
  disconnect and cancel generation. Standard HTTP clients do not do this.
- **Word-level BPE streaming.** With the word-level tokenizer, streamed text is
  emitted at token boundaries and spacing can differ slightly from a single
  final decode; the byte-level BPE tokenizer streams exactly.
- **No TLS** in-process; use a reverse proxy.
- One process serves all models; there is no multi-process worker pool.
