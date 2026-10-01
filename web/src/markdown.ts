const SAFE_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

function safeHref(raw: string): string | null {
  try {
    const url = new URL(raw, window.location.href);
    return SAFE_PROTOCOLS.has(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

const INLINE = /(`+)([\s\S]*?[^`])\1(?!`)|\*\*([^*]+?)\*\*|__([^_]+?)__|\*([^*\s][^*]*?)\*|_([^_\s][^_]*?)_|\[([^\]]+)\]\(([^)\s]+)\)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g;

function renderInline(text: string, parent: Node, depth = 0): void {
  if (depth > 4) {
    parent.appendChild(document.createTextNode(text));
    return;
  }
  let last = 0;
  INLINE.lastIndex = 0;
  const matches = Array.from(text.matchAll(INLINE));
  for (const m of matches) {
    const idx = m.index ?? 0;
    if (idx > last) parent.appendChild(document.createTextNode(text.slice(last, idx)));
    last = idx + m[0].length;

    if (m[2] !== undefined) {
      const code = el("code", "md-inline-code");
      code.textContent = m[2].replace(/^ (.*) $/, "$1");
      parent.appendChild(code);
    } else if (m[3] !== undefined || m[4] !== undefined) {
      const strong = el("strong");
      renderInline(m[3] ?? m[4], strong, depth + 1);
      parent.appendChild(strong);
    } else if (m[5] !== undefined || m[6] !== undefined) {
      const em = el("em");
      renderInline(m[5] ?? m[6], em, depth + 1);
      parent.appendChild(em);
    } else if (m[7] !== undefined) {
      const href = safeHref(m[8]);
      if (href) {
        const a = el("a");
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener noreferrer nofollow";
        renderInline(m[7], a, depth + 1);
        parent.appendChild(a);
      } else {
        parent.appendChild(document.createTextNode(m[0]));
      }
    } else if (m[9] !== undefined) {
      const href = safeHref(m[9]);
      if (href) {
        const a = el("a");
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener noreferrer nofollow";
        a.textContent = m[9];
        parent.appendChild(a);
      } else {
        parent.appendChild(document.createTextNode(m[0]));
      }
    }
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

function codeBlock(code: string, lang: string, open: boolean): HTMLElement {
  const wrap = el("div", "md-code");
  const bar = el("div", "md-code-bar");
  const label = el("span", "md-code-lang");
  label.textContent = lang || "text";
  const copy = el("button", "md-code-copy");
  copy.type = "button";
  copy.textContent = "Copy";
  copy.addEventListener("click", () => {
    navigator.clipboard?.writeText(code).then(
      () => { copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1200); },
      () => { copy.textContent = "Copy failed"; },
    );
  });
  bar.append(label, copy);
  const pre = el("pre");
  const c = el("code");
  if (lang) c.dataset.lang = lang;
  c.textContent = code;
  pre.appendChild(c);
  wrap.append(bar, pre);
  if (open) wrap.classList.add("md-code-open");
  return wrap;
}

interface ListFrame { node: HTMLOListElement | HTMLUListElement; indent: number; ordered: boolean }

export function renderMarkdown(src: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  let para: string[] = [];
  let lists: ListFrame[] = [];
  let quote: string[] | null = null;

  const flushPara = () => {
    if (!para.length) return;
    const p = el("p");
    para.forEach((line, i) => {
      if (i > 0) p.appendChild(el("br"));
      renderInline(line, p);
    });
    frag.appendChild(p);
    para = [];
  };
  const flushLists = () => { lists = []; };
  const flushQuote = () => {
    if (quote === null) return;
    const bq = el("blockquote");
    bq.appendChild(renderMarkdown(quote.join("\n")));
    frag.appendChild(bq);
    quote = null;
  };
  const flushAll = () => { flushPara(); flushLists(); flushQuote(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const fence = /^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/.exec(line);
    if (fence) {
      flushAll();
      const marker = fence[1];
      const body: string[] = [];
      let closed = false;
      for (i = i + 1; i < lines.length; i++) {
        if (lines[i].trim().startsWith(marker[0].repeat(marker.length)) && lines[i].trim().replace(/[`~]/g, "") === "") {
          closed = true;
          break;
        }
        body.push(lines[i]);
      }
      frag.appendChild(codeBlock(body.join("\n"), fence[2], !closed));
      continue;
    }

    if (/^\s*>/.test(line)) {
      flushPara(); flushLists();
      (quote ??= []).push(line.replace(/^\s*>\s?/, ""));
      continue;
    } else if (quote !== null) {
      flushQuote();
    }

    if (line.trim() === "") { flushPara(); flushLists(); continue; }

    const heading = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) {
      flushAll();
      const h = el(`h${heading[1].length}` as "h1");
      renderInline(heading[2], h);
      frag.appendChild(h);
      continue;
    }

    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flushAll();
      frag.appendChild(el("hr"));
      continue;
    }

    const item = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/.exec(line);
    if (item) {
      flushPara();
      const indent = item[1].replace(/\t/g, "    ").length;
      const ordered = /\d/.test(item[2]);
      while (lists.length && lists[lists.length - 1].indent > indent) lists.pop();
      let top = lists[lists.length - 1];
      if (!top || top.indent < indent || top.ordered !== ordered) {
        if (top && top.indent === indent) lists.pop();
        const list = ordered ? el("ol") : el("ul");
        if (ordered) {
          const start = parseInt(item[2], 10);
          if (start !== 1) (list as HTMLOListElement).start = start;
        }
        const parent = lists[lists.length - 1];
        if (parent && parent.node.lastElementChild) parent.node.lastElementChild.appendChild(list);
        else frag.appendChild(list);
        top = { node: list, indent, ordered };
        lists.push(top);
      }
      const li = el("li");
      renderInline(item[3], li);
      top.node.appendChild(li);
      continue;
    }

    if (lists.length && /^\s+\S/.test(line)) {
      const li = lists[lists.length - 1].node.lastElementChild;
      if (li) {
        li.appendChild(document.createTextNode(" "));
        renderInline(line.trim(), li);
        continue;
      }
    }

    flushLists();
    para.push(line);
  }
  flushAll();
  return frag;
}
