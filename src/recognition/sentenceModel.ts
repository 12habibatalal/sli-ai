// The phone sentence model (public/models/sentences-small.onnx): SF1 features -> glosses.

import * as ort from 'onnxruntime-web/wasm';
import { greedyCtc } from './ctc';
import { SF_DIM } from './sentenceFeatures';

export interface SentenceVocab {
  glosses: string[];
  lookup: Record<string, string>;
}

/** Gloss ids (1-based, blank = 0) -> glosses and the written sentence when it is a known one. */
export function glossText(v: SentenceVocab, ids: number[]): { glosses: string[]; text: string } {
  const glosses = ids.map((i) => v.glosses[i - 1]).filter((g): g is string => !!g);
  const key = glosses.join(' ');
  return { glosses, text: v.lookup[key] ?? key };
}

export class SentenceModel {
  private constructor(private session: ort.InferenceSession) {}

  static async load(base: string): Promise<SentenceModel> {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(`${base}models/sentences-small.onnx`, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new SentenceModel(session);
  }

  /** `feats` is T * SF_DIM values at 15 fps. */
  async run(feats: Float32Array, T: number): Promise<{ ids: number[]; conf: number[] }> {
    const out = await this.session.run({ feats: new ort.Tensor('float32', feats, [1, T, SF_DIM]) });
    const lp = out.logprobs;
    return greedyCtc(lp.data as Float32Array, lp.dims[1], lp.dims[2]);
  }
}
