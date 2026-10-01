import { readSSE } from "./sse.js";

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ModelInfo {
  id: string;
  context_length?: number;
  max_generation_tokens?: number;
  chat_template?: string;
  [key: string]: unknown;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string[];
  seed?: number;
}

export interface StreamCallbacks {
  onOpen?: () => void;
  onDelta?: (text: string) => void;
}

export interface StreamResult {
  text: string;
  finishReason: string | null;
  usage: Usage | null;
  requestId: string | null;
  firstTokenMs: number | null;
  durationMs: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null,
    readonly retryAfterS: number | null,
    readonly requestId: string | null,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get retryable(): boolean {
    return this.status === 429 || this.status === 502 || this.status === 503 || this.status === 504;
  }
}

export class NetworkError extends Error {
  constructor(message: string, readonly partial: boolean) {
    super(message);
    this.name = "NetworkError";
  }
}

export class AbortedError extends Error {
  constructor() {
    super("generation stopped");
    this.name = "AbortedError";
  }
}

export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  maxRetries?: number;
  fetchImpl?: typeof fetch;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new AbortedError()); }, { once: true });
  });

export class ChatClient {
  private baseUrl: string;
  private apiKey: string;
  private maxRetries: number;
  private fetchImpl: typeof fetch;

  constructor(opts: ClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "").replace(/\/+$/, "");
    this.apiKey = opts.apiKey ?? "";
    this.maxRetries = opts.maxRetries ?? 3;
    this.fetchImpl = opts.fetchImpl ?? fetch.bind(globalThis);
  }

  configure(opts: ClientOptions): void {
    if (opts.baseUrl !== undefined) this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    if (opts.apiKey !== undefined) this.apiKey = opts.apiKey;
    if (opts.maxRetries !== undefined) this.maxRetries = opts.maxRetries;
  }

  private headers(accept: "json" | "stream", hasBody: boolean): Record<string, string> {
    const h: Record<string, string> = { Accept: accept === "json" ? "application/json" : "text/event-stream" };
    if (hasBody) h["Content-Type"] = "application/json";
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  private static async toError(res: Response): Promise<ApiError> {
    let message = `HTTP ${res.status}`;
    let code: string | null = null;
    try {
      const body = await res.json();
      if (body && body.error) {
        message = String(body.error.message ?? message);
        code = body.error.code ?? null;
      }
    } catch {
      message = res.statusText || message;
    }
    const ra = res.headers.get("Retry-After");
    const retryAfter = ra !== null && /^\d+$/.test(ra) ? Number(ra) : null;
    return new ApiError(message, res.status, code, retryAfter, res.headers.get("X-Request-Id"));
  }

  private async send(path: string, init: RequestInit, signal?: AbortSignal): Promise<Response> {
    let attempt = 0;
    while (true) {
      let res: Response;
      try {
        res = await this.fetchImpl(this.baseUrl + path, { ...init, signal });
      } catch (err) {
        if (signal?.aborted) throw new AbortedError();
        if (attempt >= this.maxRetries) throw new NetworkError("could not reach the server", false);
        await sleep(Math.min(8000, 500 * 2 ** attempt), signal);
        attempt++;
        continue;
      }
      if (res.ok) return res;

      const error = await ChatClient.toError(res);
      if (!error.retryable || attempt >= this.maxRetries) throw error;
      const wait = error.retryAfterS !== null ? error.retryAfterS * 1000 : Math.min(8000, 500 * 2 ** attempt);
      await sleep(wait, signal);
      attempt++;
    }
  }

  async listModels(signal?: AbortSignal): Promise<ModelInfo[]> {
    const res = await this.send("/v1/models", { method: "GET", headers: this.headers("json", false) }, signal);
    const body = await res.json();
    return Array.isArray(body?.data) ? body.data : [];
  }

  async streamChat(req: ChatRequest, cb: StreamCallbacks = {}, signal?: AbortSignal): Promise<StreamResult> {
    const started = performance.now();
    const payload = { ...req, stream: true, stream_options: { include_usage: true } };
    const res = await this.send(
      "/v1/chat/completions",
      { method: "POST", headers: this.headers("stream", true), body: JSON.stringify(payload) },
      signal,
    );
    if (!res.body) throw new NetworkError("response has no body", false);
    cb.onOpen?.();

    const requestId = res.headers.get("X-Request-Id");
    let text = "";
    let finishReason: string | null = null;
    let usage: Usage | null = null;
    let firstTokenMs: number | null = null;
    let sawDone = false;

    try {
      for await (const msg of readSSE(res.body, signal)) {
        if (msg.data === "[DONE]") { sawDone = true; break; }
        let chunk: any;
        try { chunk = JSON.parse(msg.data); } catch { continue; }

        if (chunk.error) {
          throw new ApiError(String(chunk.error.message ?? "generation failed"), 500,
            chunk.error.code ?? null, null, requestId);
        }
        if (chunk.usage) usage = chunk.usage as Usage;

        const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : undefined;
        if (!choice) continue;
        const delta: string | undefined = choice.delta?.content;
        if (delta) {
          if (firstTokenMs === null) firstTokenMs = performance.now() - started;
          text += delta;
          cb.onDelta?.(delta);
        }
        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
    } catch (err) {
      if (signal?.aborted) throw new AbortedError();
      if (err instanceof ApiError) throw err;
      throw new NetworkError("connection lost during generation", text.length > 0);
    }

    if (signal?.aborted) throw new AbortedError();
    if (!sawDone) throw new NetworkError("stream ended unexpectedly", text.length > 0);

    return { text, finishReason, usage, requestId, firstTokenMs, durationMs: performance.now() - started };
  }
}
