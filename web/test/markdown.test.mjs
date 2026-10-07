import { JSDOM } from "jsdom";
import assert from "node:assert/strict";

const dom = new JSDOM("<!doctype html><body></body>", { url: "https://chat.example/" });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { renderMarkdown } = await import("../dist/markdown.js");

const html = (src) => {
  const div = document.createElement("div");
  div.appendChild(renderMarkdown(src));
  return div;
};

const attacks = [
  "<script>alert(1)</script>",
  "<img pytensorforge=x onerror=alert(1)>",
  "[click](javascript:alert(1))",
  "[click](JaVaScRiPt:alert(1))",
  "[x](data:text/html,<script>alert(1)</script>)",
  "**<svg onload=alert(1)>**",
  "`<b>code</b>`",
  "```html\n<script>alert(1)</script>\n```",
  "> <iframe pytensorforge=javascript:alert(1)>",
  "- <a href=javascript:alert(1)>x</a>",
];
for (const src of attacks) {
  const d = html(src);
  assert.equal(d.querySelectorAll("script,img,svg,iframe,object,embed").length, 0, `element injected: ${src}`);
  for (const el of d.querySelectorAll("*")) {
    for (const attr of el.getAttributeNames()) {
      assert.ok(!attr.startsWith("on"), `event handler attribute from: ${src}`);
    }
  }
  for (const a of d.querySelectorAll("a")) {
    assert.match(a.href, /^(https?|mailto):/, `unsafe href ${a.href} from: ${src}`);
    assert.equal(a.rel, "noopener noreferrer nofollow");
  }
}

let d = html("# Title\n\nSome **bold** and *em* and `code`.\n\n- a\n- b\n  - nested\n\n1. one\n2. two\n\n```python\nprint('hi')\n```\n\n> quote\n\n---\n\n[site](https://example.com) and https://auto.example/path.");
assert.equal(d.querySelector("h1").textContent, "Title");
assert.equal(d.querySelector("strong").textContent, "bold");
assert.equal(d.querySelector("em").textContent, "em");
assert.equal(d.querySelector(".md-inline-code").textContent, "code");
assert.equal(d.querySelectorAll("ul > li").length, 3);
assert.equal(d.querySelector("ul ul li").textContent, "nested");
assert.equal(d.querySelectorAll("ol > li").length, 2);
assert.equal(d.querySelector(".md-code code").textContent, "print('hi')");
assert.equal(d.querySelector(".md-code-lang").textContent, "python");
assert.ok(d.querySelector("blockquote"));
assert.ok(d.querySelector("hr"));
assert.deepEqual([...d.querySelectorAll("a")].map((a) => a.href), ["https://example.com/", "https://auto.example/path"]);

d = html("```js\nconst x = 1;\nconst y");
assert.ok(d.querySelector(".md-code-open"), "unterminated fence renders as open code block while streaming");
assert.equal(d.querySelector(".md-code code").textContent, "const x = 1;\nconst y");

d = html("*".repeat(5000) + "a");
assert.ok(d.textContent.length > 0, "pathological emphasis input terminates");

console.log("markdown renderer: XSS vectors neutralized, formatting correct, streaming-safe");
