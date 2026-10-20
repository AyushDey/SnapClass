import { GlobalWindow } from "happy-dom";

const win = new GlobalWindow({ url: "http://localhost:5173" });
for (const p of Object.getOwnPropertyNames(win)) {
  if (!(p in globalThis)) {
    try {
      globalThis[p] = win[p];
    } catch {
      // ignore read-only properties
    }
  }
}
globalThis.window = win;
globalThis.document = win.document;
globalThis.navigator = win.navigator;
