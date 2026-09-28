// The large sentence model on the SLI server (server/sentence-api). Only SF1 numbers are sent,
// never camera images. Any failure, including a slow answer, returns null so the caller can
// fall back to the phone model.

export interface ServerResult {
  ids: number[];
  glosses: string[];
  text: string;
  conf: number[];
  ms: number;
  model: string;
}

export async function recognizeOnServer(
  feats: Float32Array,
  opts: { url?: string; timeoutMs?: number; fetchImpl?: typeof fetch } = {},
): Promise<ServerResult | null> {
  const { url = '/api/sentence', timeoutMs = 4000, fetchImpl = fetch } = opts;
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return null;
  const ctrl = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => {
      ctrl.abort();
      resolve(null);
    }, timeoutMs);
  });
  const request = (async () => {
    try {
      const body = feats.buffer.slice(feats.byteOffset, feats.byteOffset + feats.byteLength) as ArrayBuffer;
      const res = await fetchImpl(url, {
        method: 'POST',
        body,
        signal: ctrl.signal,
        headers: { 'Content-Type': 'application/octet-stream' },
      });
      return res.ok ? ((await res.json()) as ServerResult) : null;
    } catch {
      return null;
    }
  })();
  try {
    return await Promise.race([request, timeout]);
  } finally {
    clearTimeout(timer);
  }
}
