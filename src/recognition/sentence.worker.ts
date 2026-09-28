/// <reference lib="webworker" />
// The phone sentence model runs in its own worker, like the word model, so a decode never
// delays the next camera frame.

import { SentenceModel } from './sentenceModel';

export type SentenceWorkerRequest = { type: 'init'; base: string } | { type: 'run'; id: number; feats: Float32Array; T: number };

export type SentenceWorkerReply =
  | { type: 'ready' }
  | { type: 'error'; message: string; id?: number }
  | { type: 'result'; id: number; ids: number[]; conf: number[]; ms: number };

let model: SentenceModel | null = null;
const post = (m: SentenceWorkerReply) => self.postMessage(m);

self.onmessage = async (e: MessageEvent<SentenceWorkerRequest>) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      model = await SentenceModel.load(m.base);
      post({ type: 'ready' });
    } else {
      const start = performance.now();
      const r = await model!.run(m.feats, m.T);
      post({ type: 'result', id: m.id, ...r, ms: performance.now() - start });
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err), id: m.type === 'run' ? m.id : undefined });
  }
};
