import type { Role, Usage } from "./api.js";

export type MessageStatus = "done" | "streaming" | "error" | "stopped";

export interface StoredMessage {
  id: string;
  role: Role;
  content: string;
  status: MessageStatus;
  createdAt: number;
  error?: string;
  retryable?: boolean;
  usage?: Usage | null;
  finishReason?: string | null;
  model?: string;
  ttftMs?: number | null;
  durationMs?: number;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  model?: string;
  messages: StoredMessage[];
}

export interface Settings {
  baseUrl: string;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  rememberKey: boolean;
}

const CONV_KEY = "ptf-chat:conversations:v1";
const SETTINGS_KEY = "ptf-chat:settings:v1";
const API_KEY = "ptf-chat:api-key";
const MAX_CONVERSATIONS = 200;

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: "",
  model: "",
  temperature: 0.8,
  maxTokens: 512,
  systemPrompt: "",
  rememberKey: false,
};

export function uid(): string {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function readJSON<T>(storage: Storage, key: string, fallback: T): T {
  try {
    const raw = storage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export class ConversationStore {
  private conversations: Conversation[];
  private saveTimer: number | null = null;

  constructor(private storage: Storage = localStorage) {
    const loaded = readJSON<Conversation[]>(storage, CONV_KEY, []);
    this.conversations = Array.isArray(loaded) ? loaded.filter((c) => c && Array.isArray(c.messages)) : [];
    for (const c of this.conversations) {
      for (const m of c.messages) {
        if (m.status === "streaming") {
          m.status = m.content ? "stopped" : "error";
          if (!m.content) m.error = "interrupted";
        }
      }
    }
  }

  list(): Conversation[] {
    return [...this.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  get(id: string): Conversation | undefined {
    return this.conversations.find((c) => c.id === id);
  }

  create(model?: string): Conversation {
    const now = Date.now();
    const conv: Conversation = { id: uid(), title: "New chat", createdAt: now, updatedAt: now, model, messages: [] };
    this.conversations.push(conv);
    this.enforceLimit();
    this.save();
    return conv;
  }

  remove(id: string): void {
    this.conversations = this.conversations.filter((c) => c.id !== id);
    this.save(true);
  }

  touch(conv: Conversation): void {
    conv.updatedAt = Date.now();
    if (conv.title === "New chat") {
      const first = conv.messages.find((m) => m.role === "user");
      if (first) conv.title = first.content.trim().replace(/\s+/g, " ").slice(0, 60) || "New chat";
    }
    this.save();
  }

  private enforceLimit(): void {
    if (this.conversations.length <= MAX_CONVERSATIONS) return;
    this.conversations = this.list().slice(0, MAX_CONVERSATIONS);
  }

  save(immediate = false): void {
    if (this.saveTimer !== null) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    const write = () => {
      this.saveTimer = null;
      for (let attempt = 0; attempt < 5; attempt++) {
        try {
          this.storage.setItem(CONV_KEY, JSON.stringify(this.conversations));
          return;
        } catch {
          const ordered = this.list();
          if (ordered.length <= 1) return;
          const victim = ordered[ordered.length - 1];
          this.conversations = this.conversations.filter((c) => c.id !== victim.id);
        }
      }
    };
    if (immediate) write();
    else this.saveTimer = window.setTimeout(write, 250);
  }
}

export function loadSettings(): Settings {
  return { ...DEFAULT_SETTINGS, ...readJSON<Partial<Settings>>(localStorage, SETTINGS_KEY, {}) };
}

export function saveSettings(s: Settings): void {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { }
}

export function loadApiKey(): string {
  return sessionStorage.getItem(API_KEY) ?? localStorage.getItem(API_KEY) ?? "";
}

export function saveApiKey(key: string, remember: boolean): void {
  try {
    sessionStorage.setItem(API_KEY, key);
    if (remember) localStorage.setItem(API_KEY, key);
    else localStorage.removeItem(API_KEY);
  } catch { }
}
