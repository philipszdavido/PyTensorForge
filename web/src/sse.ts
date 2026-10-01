export interface SSEMessage {
  event: string;
  data: string;
  id?: string;
}

export async function* readSSE(body: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncGenerator<SSEMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let dataLines: string[] = [];
  let eventName = "message";
  let lastId: string | undefined;

  const onAbort = () => {
    reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
      } else {
        buffer += decoder.decode(value, { stream: true });
      }

      let start = 0;
      while (true) {
        let end = -1;
        let skip = 1;
        for (let i = start; i < buffer.length; i++) {
          const c = buffer.charCodeAt(i);
          if (c === 10) { end = i; skip = 1; break; }
          if (c === 13) {
            if (i + 1 >= buffer.length && !done) { end = -1; break; }
            end = i;
            skip = buffer.charCodeAt(i + 1) === 10 ? 2 : 1;
            break;
          }
        }
        if (end < 0) break;

        const line = buffer.slice(start, end);
        start = end + skip;

        if (line === "") {
          if (dataLines.length > 0) {
            yield { event: eventName, data: dataLines.join("\n"), id: lastId };
          }
          dataLines = [];
          eventName = "message";
          continue;
        }
        if (line.startsWith(":")) continue;

        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        let val = colon < 0 ? "" : line.slice(colon + 1);
        if (val.startsWith(" ")) val = val.slice(1);

        if (field === "data") dataLines.push(val);
        else if (field === "event") eventName = val || "message";
        else if (field === "id" && !val.includes("\u0000")) lastId = val;
      }
      buffer = buffer.slice(start);

      if (done) {
        if (dataLines.length > 0) yield { event: eventName, data: dataLines.join("\n"), id: lastId };
        return;
      }
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
}
