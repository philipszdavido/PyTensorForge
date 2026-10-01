import { AbortedError, ApiError, ChatClient, ChatMessage, ModelInfo, NetworkError } from "./api.js";
import { renderMarkdown } from "./markdown.js";
import {
  Conversation, ConversationStore, Settings, StoredMessage,
  loadApiKey, loadSettings, saveApiKey, saveSettings, uid,
} from "./store.js";

type GenState = "idle" | "connecting" | "streaming" | "error";

function $(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
}

class ChatApp {
  private store = new ConversationStore();
  private settings: Settings = loadSettings();
  private client = new ChatClient({ baseUrl: this.settings.baseUrl, apiKey: loadApiKey() });
  private models: ModelInfo[] = [];
  private current: Conversation | null = null;
  private controller: AbortController | null = null;
  private state: GenState = "idle";
  private renderQueued = false;
  private streamingNode: HTMLElement | null = null;
  private streamingMsg: StoredMessage | null = null;
  private streamChunks = 0;
  private streamStarted = 0;

  private els = {
    list: $("conversations"),
    messages: $("messages"),
    input: $("input") as HTMLTextAreaElement,
    send: $("send") as HTMLButtonElement,
    stop: $("stop") as HTMLButtonElement,
    model: $("model") as HTMLSelectElement,
    status: $("status-line"),
    conn: $("conn-status"),
    banner: $("banner"),
    dialog: $("settings") as HTMLDialogElement,
  };

  start(): void {
    $("new-chat").addEventListener("click", () => this.newConversation());
    this.els.send.addEventListener("click", () => this.send());
    this.els.stop.addEventListener("click", () => this.stopGeneration());
    this.els.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.send();
      }
    });
    this.els.input.addEventListener("input", () => this.autosize());
    this.els.model.addEventListener("change", () => {
      this.settings.model = this.els.model.value;
      saveSettings(this.settings);
      if (this.current) { this.current.model = this.settings.model; this.store.touch(this.current); }
    });
    $("settings-btn").addEventListener("click", () => this.openSettings());
    $("settings-save").addEventListener("click", () => this.saveSettingsFromDialog());
    $("settings-cancel").addEventListener("click", () => this.els.dialog.close());
    window.addEventListener("online", () => { this.showBanner(null); this.refreshModels(); });
    window.addEventListener("offline", () => this.showBanner("You are offline. Messages will fail until the connection returns."));
    window.addEventListener("beforeunload", () => this.store.save(true));

    const first = this.store.list()[0];
    if (first) this.select(first.id); else this.newConversation();
    this.renderList();
    this.refreshModels();
    this.setState("idle");
  }

  private showBanner(text: string | null): void {
    this.els.banner.textContent = text ?? "";
    this.els.banner.hidden = !text;
  }

  private async refreshModels(): Promise<void> {
    this.els.conn.textContent = "connecting…";
    this.els.conn.className = "conn conn-pending";
    try {
      this.models = await this.client.listModels();
      this.els.conn.textContent = "connected";
      this.els.conn.className = "conn conn-ok";
      this.showBanner(null);
    } catch (err) {
      this.models = [];
      this.els.conn.textContent = "disconnected";
      this.els.conn.className = "conn conn-bad";
      if (err instanceof ApiError && err.status === 401) {
        this.showBanner("The server requires an API key. Open Settings to add one.");
      } else {
        this.showBanner(`Cannot load models: ${(err as Error).message}`);
      }
    }
    this.renderModels();
  }

  private renderModels(): void {
    const sel = this.els.model;
    sel.replaceChildren();
    for (const m of this.models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.context_length ? `${m.id} (${m.context_length} ctx)` : m.id;
      sel.appendChild(opt);
    }
    const wanted = this.current?.model || this.settings.model;
    if (wanted && this.models.some((m) => m.id === wanted)) sel.value = wanted;
    else if (this.models[0]) sel.value = this.models[0].id;
    sel.disabled = this.models.length === 0;
  }

  private newConversation(): void {
    if (this.state === "streaming" || this.state === "connecting") this.stopGeneration();
    const conv = this.store.create(this.els.model.value || this.settings.model);
    this.select(conv.id);
    this.renderList();
    this.els.input.focus();
  }

  private select(id: string): void {
    if (this.current?.id !== id && (this.state === "streaming" || this.state === "connecting")) this.stopGeneration();
    this.current = this.store.get(id) ?? null;
    if (this.current?.model && this.models.some((m) => m.id === this.current!.model)) this.els.model.value = this.current.model;
    this.renderList();
    this.renderMessages();
  }

  private renderList(): void {
    const ul = this.els.list;
    ul.replaceChildren();
    for (const conv of this.store.list()) {
      const li = document.createElement("li");
      li.className = conv.id === this.current?.id ? "conv active" : "conv";
      const title = document.createElement("button");
      title.type = "button";
      title.className = "conv-title";
      title.textContent = conv.title;
      title.addEventListener("click", () => this.select(conv.id));
      const del = document.createElement("button");
      del.type = "button";
      del.className = "conv-delete";
      del.title = "Delete conversation";
      del.setAttribute("aria-label", "Delete conversation");
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${conv.title}"?`)) return;
        const wasCurrent = conv.id === this.current?.id;
        if (wasCurrent && this.state !== "idle" && this.state !== "error") this.stopGeneration();
        this.store.remove(conv.id);
        if (wasCurrent) {
          const next = this.store.list()[0];
          if (next) this.select(next.id); else this.newConversation();
        }
        this.renderList();
      });
      li.append(title, del);
      ul.appendChild(li);
    }
  }

  private messageNode(msg: StoredMessage, isLast: boolean): HTMLElement {
    const wrap = document.createElement("article");
    wrap.className = `msg msg-${msg.role} msg-${msg.status}`;
    wrap.dataset.id = msg.id;

    const role = document.createElement("div");
    role.className = "msg-role";
    role.textContent = msg.role === "user" ? "You" : msg.role === "assistant" ? (msg.model || "Assistant") : "System";

    const body = document.createElement("div");
    body.className = "msg-body";
    if (msg.role === "assistant") body.appendChild(renderMarkdown(msg.content));
    else body.textContent = msg.content;

    wrap.append(role, body);

    if (msg.status === "streaming" && !msg.content) {
      const dots = document.createElement("span");
      dots.className = "typing";
      dots.textContent = "…";
      body.appendChild(dots);
    }

    if (msg.role === "assistant" && msg.status !== "streaming") {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      const parts: string[] = [];
      if (msg.status === "stopped") parts.push("stopped");
      if (msg.finishReason === "length") parts.push("reached token limit");
      if (msg.usage) parts.push(`${msg.usage.completion_tokens} tokens`);
      if (msg.usage && msg.durationMs && msg.ttftMs != null && msg.usage.completion_tokens > 1) {
        const rate = (msg.usage.completion_tokens - 1) / Math.max(0.001, (msg.durationMs - msg.ttftMs) / 1000);
        parts.push(`${rate.toFixed(1)} tok/s`);
      }
      if (msg.ttftMs != null) parts.push(`TTFT ${Math.round(msg.ttftMs)} ms`);
      meta.textContent = parts.join(" · ");

      if (msg.status === "error") {
        const err = document.createElement("div");
        err.className = "msg-error";
        err.textContent = msg.error || "generation failed";
        wrap.appendChild(err);
      }

      if (isLast) {
        const actions = document.createElement("div");
        actions.className = "msg-actions";
        const again = document.createElement("button");
        again.type = "button";
        again.textContent = msg.status === "error" ? "Retry" : "Regenerate";
        again.addEventListener("click", () => this.regenerate());
        actions.appendChild(again);
        if (msg.content) {
          const copy = document.createElement("button");
          copy.type = "button";
          copy.textContent = "Copy";
          copy.addEventListener("click", () => navigator.clipboard?.writeText(msg.content));
          actions.appendChild(copy);
        }
        meta.appendChild(actions);
      }
      wrap.appendChild(meta);
    }
    return wrap;
  }

  private renderMessages(): void {
    const box = this.els.messages;
    box.replaceChildren();
    this.streamingNode = null;
    const msgs = this.current?.messages ?? [];
    if (!msgs.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "Start a conversation.";
      box.appendChild(empty);
      return;
    }
    msgs.forEach((m, i) => {
      const node = this.messageNode(m, i === msgs.length - 1);
      if (m === this.streamingMsg) this.streamingNode = node;
      box.appendChild(node);
    });
    this.scrollToEnd(true);
  }

  private scrollToEnd(force = false): void {
    const box = this.els.messages;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
    if (force || nearBottom) box.scrollTop = box.scrollHeight;
  }

  private queueStreamRender(): void {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      if (!this.streamingMsg || !this.streamingNode) return;
      const body = this.streamingNode.querySelector(".msg-body");
      if (body) body.replaceChildren(renderMarkdown(this.streamingMsg.content));
      this.scrollToEnd();
      this.updateStatus();
    });
  }

  private setState(state: GenState): void {
    this.state = state;
    const busy = state === "connecting" || state === "streaming";
    this.els.send.hidden = busy;
    this.els.stop.hidden = !busy;
    this.els.input.disabled = false;
    this.updateStatus();
  }

  private updateStatus(): void {
    const s = this.els.status;
    if (this.state === "connecting") s.textContent = "Waiting for the model…";
    else if (this.state === "streaming") {
      const secs = (performance.now() - this.streamStarted) / 1000;
      s.textContent = `Generating · ~${this.streamChunks} tokens · ${secs.toFixed(1)} s`;
    } else if (this.state === "error") s.textContent = "Last request failed.";
    else s.textContent = "";
  }

  private autosize(): void {
    const t = this.els.input;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 240)}px`;
  }

  private history(conv: Conversation, upTo: number): ChatMessage[] {
    const out: ChatMessage[] = [];
    if (this.settings.systemPrompt.trim()) out.push({ role: "system", content: this.settings.systemPrompt });
    for (const m of conv.messages.slice(0, upTo)) {
      if (m.role === "assistant" && (m.status === "error" || !m.content)) continue;
      out.push({ role: m.role, content: m.content });
    }
    return out;
  }

  private send(): void {
    const text = this.els.input.value.trim();
    if (!text || !this.current || this.state === "streaming" || this.state === "connecting") return;
    if (!this.els.model.value) {
      this.showBanner("No model is available. Check the connection and Settings.");
      return;
    }
    this.els.input.value = "";
    this.autosize();
    this.current.messages.push({ id: uid(), role: "user", content: text, status: "done", createdAt: Date.now() });
    this.store.touch(this.current);
    this.renderList();
    this.generate();
  }

  private regenerate(): void {
    const conv = this.current;
    if (!conv || this.state === "streaming" || this.state === "connecting") return;
    const last = conv.messages[conv.messages.length - 1];
    if (last?.role === "assistant") conv.messages.pop();
    if (!conv.messages.length || conv.messages[conv.messages.length - 1].role !== "user") return;
    this.generate();
  }

  private stopGeneration(): void {
    this.controller?.abort();
  }

  private async generate(): Promise<void> {
    const conv = this.current!;
    const model = this.els.model.value;
    const messages = this.history(conv, conv.messages.length);
    const msg: StoredMessage = {
      id: uid(), role: "assistant", content: "", status: "streaming", createdAt: Date.now(), model,
    };
    conv.messages.push(msg);
    this.streamingMsg = msg;
    this.streamChunks = 0;
    this.streamStarted = performance.now();
    this.renderMessages();

    const controller = new AbortController();
    this.controller = controller;
    this.setState("connecting");

    try {
      const result = await this.client.streamChat(
        { model, messages, temperature: this.settings.temperature, max_tokens: this.settings.maxTokens },
        {
          onOpen: () => this.setState("streaming"),
          onDelta: (delta) => {
            msg.content += delta;
            this.streamChunks++;
            this.queueStreamRender();
          },
        },
        controller.signal,
      );
      msg.content = result.text;
      msg.status = "done";
      msg.usage = result.usage;
      msg.finishReason = result.finishReason;
      msg.ttftMs = result.firstTokenMs;
      msg.durationMs = result.durationMs;
      this.finish(conv, "idle");
    } catch (err) {
      if (err instanceof AbortedError) {
        msg.status = msg.content ? "stopped" : "error";
        if (!msg.content) msg.error = "stopped before any output";
        this.finish(conv, "idle");
      } else {
        msg.status = "error";
        msg.error = describeError(err);
        this.finish(conv, "error");
        if (err instanceof NetworkError && !navigator.onLine) this.showBanner("You are offline.");
        if (err instanceof ApiError && err.status === 401) this.showBanner("Invalid or missing API key. Open Settings.");
      }
    }
  }

  private finish(conv: Conversation, state: GenState): void {
    this.controller = null;
    this.streamingMsg = null;
    this.store.touch(conv);
    this.setState(state);
    if (conv === this.current) this.renderMessages();
    this.renderList();
  }

  private openSettings(): void {
    const f = (id: string) => $(id) as HTMLInputElement;
    f("set-base-url").value = this.settings.baseUrl;
    f("set-api-key").value = loadApiKey();
    f("set-remember").checked = this.settings.rememberKey;
    f("set-temperature").value = String(this.settings.temperature);
    f("set-max-tokens").value = String(this.settings.maxTokens);
    ($("set-system") as HTMLTextAreaElement).value = this.settings.systemPrompt;
    this.els.dialog.showModal();
  }

  private saveSettingsFromDialog(): void {
    const f = (id: string) => $(id) as HTMLInputElement;
    const temperature = Number(f("set-temperature").value);
    const maxTokens = Math.floor(Number(f("set-max-tokens").value));
    this.settings = {
      ...this.settings,
      baseUrl: f("set-base-url").value.trim(),
      rememberKey: f("set-remember").checked,
      temperature: Number.isFinite(temperature) ? Math.min(2, Math.max(0, temperature)) : this.settings.temperature,
      maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : this.settings.maxTokens,
      systemPrompt: ($("set-system") as HTMLTextAreaElement).value,
    };
    saveSettings(this.settings);
    const key = f("set-api-key").value.trim();
    saveApiKey(key, this.settings.rememberKey);
    this.client.configure({ baseUrl: this.settings.baseUrl, apiKey: key });
    this.els.dialog.close();
    this.refreshModels();
  }
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) return `Rate limited: ${err.message}`;
    if (err.status === 503) return `Server busy: ${err.message}`;
    return err.requestId ? `${err.message} (request ${err.requestId})` : err.message;
  }
  if (err instanceof NetworkError) {
    return err.partial ? "Connection lost mid-response. Retry to regenerate." : "Could not reach the server.";
  }
  return "Unexpected error.";
}

new ChatApp().start();
