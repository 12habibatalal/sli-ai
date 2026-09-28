// Sentences mode: buffer one stretch of signing (from the first visible hand until the hands
// have been down for endGapMs, or maxMs), show phone-model guesses while signing, then ask the
// large server model for the final sentence and fall back to the phone model.

import type { RawFrame } from './features';
import { SF_DIM, resampleFrames, sentenceFrame } from './sentenceFeatures';
import { glossText, type SentenceVocab } from './sentenceModel';
import { recognizeOnServer } from './sentenceClient';

export interface SentenceResult {
  glosses: string[];
  text: string;
  source: 'server' | 'phone';
  final: boolean;
}

export interface SentenceOptions {
  endGapMs: number; // hands down this long ends the sentence
  minSignMs: number; // shorter stretches of hand movement are ignored
  maxMs: number; // a sentence is cut here even if the hands never drop
  liveEveryMs: number; // phone-model preview interval while signing
  serverTimeoutMs: number;
}

export const DEFAULT_SENTENCE: SentenceOptions = {
  endGapMs: 1200,
  minSignMs: 600,
  maxMs: 30000,
  liveEveryMs: 1000,
  serverTimeoutMs: 4000,
};

const MAX_FRAMES = 450; // 30 s at 15 fps, the server's limit
const MIN_FRAMES = 8;

type Run = (feats: Float32Array, T: number) => Promise<{ ids: number[]; conf: number[] }>;

export class SentenceRecognizer {
  private frames: Float32Array[] = [];
  private times: number[] = [];
  private firstHand = -1;
  private lastHand = -1;
  private lastLive = -1;
  private busy = false;

  constructor(
    private runPhone: Run,
    private vocab: SentenceVocab,
    private events: { onLive?(r: SentenceResult): void; onSentence?(r: SentenceResult): void },
    private opts: SentenceOptions = DEFAULT_SENTENCE,
    private server: typeof recognizeOnServer = recognizeOnServer,
  ) {}

  reset() {
    this.frames = [];
    this.times = [];
    this.firstHand = this.lastHand = this.lastLive = -1;
  }

  /** The buffered stretch resampled to 15 fps, keeping the last MAX_FRAMES frames. */
  private window(): { feats: Float32Array; T: number } {
    let feats = resampleFrames(this.frames, this.times);
    let T = feats.length / SF_DIM;
    if (T > MAX_FRAMES) {
      feats = feats.slice((T - MAX_FRAMES) * SF_DIM);
      T = MAX_FRAMES;
    }
    return { feats, T };
  }

  async push(t: number, raw: RawFrame) {
    const seen = raw.hands.length > 0;
    if (seen) {
      if (this.firstHand < 0) this.firstHand = t;
      this.lastHand = t;
    }
    if (this.firstHand < 0) return; // nothing to buffer before the first hand
    this.frames.push(sentenceFrame(raw));
    this.times.push(t);
    const signedFor = this.lastHand - this.firstHand;
    const ended = !seen && t - this.lastHand >= this.opts.endGapMs;
    const full = t - this.firstHand >= this.opts.maxMs;
    if (ended || full) {
      const { feats, T } = this.window();
      this.reset();
      if (signedFor >= this.opts.minSignMs && T >= MIN_FRAMES) await this.finish(feats, T);
      return;
    }
    if (seen && !this.busy && signedFor >= this.opts.minSignMs && t - this.lastLive >= this.opts.liveEveryMs) {
      this.lastLive = t;
      this.busy = true;
      try {
        const { feats, T } = this.window();
        const r = await this.runPhone(feats, T);
        this.events.onLive?.({ ...glossText(this.vocab, r.ids), source: 'phone', final: false });
      } finally {
        this.busy = false;
      }
    }
  }

  private async finish(feats: Float32Array, T: number) {
    const s = await this.server(feats, { timeoutMs: this.opts.serverTimeoutMs });
    if (s) {
      this.events.onSentence?.({ glosses: s.glosses, text: s.text, source: 'server', final: true });
      return;
    }
    const r = await this.runPhone(feats, T);
    this.events.onSentence?.({ ...glossText(this.vocab, r.ids), source: 'phone', final: true });
  }
}
