import { describe, expect, it, vi } from 'vitest';
import type { RawFrame } from '../../src/recognition/features';
import { DEFAULT_SENTENCE, SentenceRecognizer, type SentenceResult } from '../../src/recognition/sentences';
import { recognizeOnServer, type ServerResult } from '../../src/recognition/sentenceClient';

const vocab = { glosses: ['a', 'b'], lookup: { 'a b': 'AB' } };

const hand = (): RawFrame => ({
  pose: Array.from({ length: 33 }, (_, i) => [0.5 + i * 0.001, 0.5, 0, 1] as [number, number, number, number]),
  face: null,
  hands: [{ label: 'Right', lms: Array.from({ length: 21 }, (_, i) => [0.5 + i * 0.01, 0.6, 0] as [number, number, number]) }],
});
const none = (): RawFrame => ({ pose: null, face: null, hands: [] });

/** Signs for `ms` at ~15 fps, then lowers the hands long enough to end the sentence. */
async function sign(r: SentenceRecognizer, ms: number, start = 0) {
  let t = start;
  for (; t < start + ms; t += 66) await r.push(t, hand());
  for (let u = t; u < t + DEFAULT_SENTENCE.endGapMs + 200; u += 66) await r.push(u, none());
}

const serverSays = (): ServerResult => ({ ids: [1, 2], glosses: ['a', 'b'], text: 'AB', conf: [0.9, 0.9], ms: 5, model: 'large-v1' });

describe('SentenceRecognizer', () => {
  it('uses the server result when it answers', async () => {
    const out: SentenceResult[] = [];
    const phone = vi.fn(async () => ({ ids: [1], conf: [0.5] }));
    const r = new SentenceRecognizer(phone, vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE, async () => serverSays());
    await sign(r, 2000);
    expect(out).toEqual([{ glosses: ['a', 'b'], text: 'AB', source: 'server', final: true }]);
  });

  it('falls back to the phone model when the server fails', async () => {
    const out: SentenceResult[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1, 2], conf: [0.8, 0.8] }), vocab, { onSentence: (s) => out.push(s) },
      DEFAULT_SENTENCE, async () => null);
    await sign(r, 2000);
    expect(out).toEqual([{ glosses: ['a', 'b'], text: 'AB', source: 'phone', final: true }]);
  });

  it('shows live phone guesses while signing', async () => {
    const live: SentenceResult[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1], conf: [0.8] }), vocab, { onLive: (s) => live.push(s) },
      DEFAULT_SENTENCE, async () => null);
    await sign(r, 3000);
    expect(live.length).toBeGreaterThan(0);
    expect(live[0]).toEqual({ glosses: ['a'], text: 'a', source: 'phone', final: false });
  });

  it('ignores movements shorter than minSignMs', async () => {
    const server = vi.fn(async () => null);
    const out: SentenceResult[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1], conf: [1] }), vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE, server);
    await sign(r, 300);
    expect(out).toHaveLength(0);
    expect(server).not.toHaveBeenCalled();
  });

  it('finalises at maxMs even if the hands never drop, sending at most 450 frames', async () => {
    const out: SentenceResult[] = [];
    const frames: number[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1], conf: [1] }), vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE,
      async (f) => {
        frames.push(f.length / 356);
        return null;
      });
    for (let t = 0; t < 31000; t += 66) await r.push(t, hand());
    expect(out).toHaveLength(1);
    expect(frames[0]).toBeLessThanOrEqual(450);
  });
});

describe('recognizeOnServer', () => {
  it('gives up after timeoutMs and returns null', async () => {
    const hang = (() => new Promise(() => {})) as unknown as typeof fetch;
    const start = Date.now();
    expect(await recognizeOnServer(new Float32Array(356 * 10), { fetchImpl: hang, timeoutMs: 100 })).toBeNull();
    expect(Date.now() - start).toBeLessThan(1000);
  });

  it('returns null on an HTTP error', async () => {
    const fail = (async () => new Response('{}', { status: 503 })) as unknown as typeof fetch;
    expect(await recognizeOnServer(new Float32Array(356 * 10), { fetchImpl: fail })).toBeNull();
  });

  it('posts the features as float32 bytes', async () => {
    let body: ArrayBuffer | null = null;
    const ok = (async (_u: string, init: RequestInit) => {
      body = init.body as ArrayBuffer;
      return new Response(JSON.stringify(serverSays()), { status: 200 });
    }) as unknown as typeof fetch;
    const feats = new Float32Array(356 * 10).fill(0.25);
    expect((await recognizeOnServer(feats, { fetchImpl: ok }))?.text).toBe('AB');
    expect(new Float32Array(body!)).toEqual(feats);
  });
});
