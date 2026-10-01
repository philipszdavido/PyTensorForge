import glob
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from test.gpt.test_phase4_server import USER_KEY, build_model_dir, check, slow_down, start_server


def find_browser():
    env = os.environ.get("PTF_BROWSER")

    if env and os.path.isfile(env):
        return env

    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "/opt/pw-browsers", os.path.expanduser("~/.cache/ms-playwright")]

    for root in roots:
        for pattern in ("chromium_headless_shell-*/chrome-linux/headless_shell", "chromium-*/chrome-linux/chrome"):
            hits = sorted(glob.glob(os.path.join(root, pattern)))

            if hits:
                return hits[-1]

    return None


def wait_until(page, expression, timeout=10.0):
    deadline = time.time() + timeout

    while time.time() < deadline:
        if page.evaluate(expression):
            return

        time.sleep(0.05)

    raise TimeoutError(f"condition not met: {expression}")


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; skipping browser test")
        return

    browser_path = find_browser()

    if browser_path is None:
        print("no chromium binary found; skipping browser test")
        return

    tmp = tempfile.mkdtemp()
    srv = None

    try:
        model_dir, _ = build_model_dir(tmp, name="ui", ctx=128, stop_on_eos=False)
        srv = start_server(model_dir, limits={"max_generation_tokens": 100})
        base = f"http://127.0.0.1:{srv.port}"
        generator = srv.models.get("tiny").generator

        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=browser_path, args=["--no-sandbox"])
            page = browser.new_page()
            problems = []
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: problems.append(str(e)))

            page.goto(base + "/")
            page.wait_for_selector("#banner:not([hidden])")
            check("API key" in page.text_content("#banner"), "UI asks for an API key on 401")
            check(page.text_content("#conn-status") == "disconnected", "connection state shown")

            page.click("#settings-btn")
            page.fill("#set-api-key", USER_KEY)
            page.fill("#set-temperature", "0")
            page.fill("#set-max-tokens", "12")
            page.click("#settings-save")
            wait_until(page, "document.querySelector('#conn-status').textContent === 'connected'")
            check(page.input_value("#model") == "tiny", "model list populated")

            page.fill("#input", "hello world")
            page.press("#input", "Enter")
            page.wait_for_selector("article.msg-assistant.msg-done", timeout=15000)
            first = page.text_content("article.msg-assistant .msg-body")
            meta = page.text_content("article.msg-assistant .msg-meta")
            check(first and "tokens" in meta and "TTFT" in meta, f"streamed reply with usage meta: {meta!r}")
            check(page.text_content("article.msg-user .msg-body") == "hello world", "user message rendered")
            check("hello world" in page.text_content("#conversations"), "conversation titled from first message")

            page.click("article.msg-assistant .msg-actions button:has-text('Regenerate')")
            page.wait_for_selector("article.msg-assistant.msg-done", timeout=15000)
            check(page.text_content("article.msg-assistant .msg-body") == first, "greedy regenerate is identical")
            check(page.locator("article.msg-assistant").count() == 1, "regenerate replaces the last reply")

            page.reload()
            wait_until(page, "document.querySelector('#conn-status').textContent === 'connected'")
            check(page.locator("article.msg").count() == 2, "conversation persisted across reload")

            slow_down(generator, 0.05)
            page.click("#settings-btn")
            page.fill("#set-max-tokens", "100")
            page.fill("#set-temperature", "0.9")
            page.click("#settings-save")
            page.fill("#input", "tell me more")
            page.press("#input", "Enter")
            page.wait_for_selector("#stop:not([hidden])")
            wait_until(page, "(() => { const b = document.querySelectorAll('article.msg-assistant')[1]"
                             "?.querySelector('.msg-body'); return !!b && !b.querySelector('.typing')"
                             " && b.textContent.length > 0; })()", timeout=15)
            page.click("#stop")
            page.wait_for_selector("article.msg-assistant.msg-stopped", timeout=10000)
            check(page.is_visible("#send") and not page.is_visible("#stop"), "controls reset after stop")
            sched = generator.scheduler
            deadline = time.time() + 5

            while time.time() < deadline and (sched.num_active or sched.cache_manager.used_bytes):
                time.sleep(0.02)

            check(sched.num_active == 0 and sched.cache_manager.used_bytes == 0, "stop in UI cancels server generation")

            page.click("#new-chat")
            check(page.locator("#conversations li").count() == 2 and page.locator("article.msg").count() == 0,
                  "new conversation")

            srv.stop()
            srv = None
            page.fill("#input", "anyone there?")
            page.press("#input", "Enter")
            page.wait_for_selector("article.msg-assistant.msg-error", timeout=20000)
            check("Could not reach the server" in page.text_content("article.msg-error .msg-error"),
                  "network failure shown to the user")
            check(page.locator("article.msg-error button:has-text('Retry')").count() == 1, "retry offered")

            benign = [m for m in problems if "Failed to load resource" not in m and "ERR_CONNECTION_REFUSED" not in m]
            check(not benign, f"no console errors or CSP violations: {benign}")
            browser.close()
    finally:
        if srv is not None:
            srv.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    print("browser: API key flow, streaming, regenerate, persistence, stop cancels server-side, offline error ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
