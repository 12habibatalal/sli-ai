// The browser path (SF1 in TypeScript -> phone ONNX model -> greedy CTC) gives what Python gives
// on the parity cases (training/sentences/export_parity.py --model).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-node';
import type { RawFrame } from '../../src/recognition/features';
import { SF_DIM, resampleFrames, sentenceFrame } from '../../src/recognition/sentenceFeatures';
import { greedyCtc } from '../../src/recognition/ctc';
import { glossText, type SentenceVocab } from '../../src/recognition/sentenceModel';

const read = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const cases: { name: string; raws: RawFrame[]; t: number[] }[] = read('../fixtures/sentence-parity.json');
const expected: { name: string; logprobs_head: number[]; ids: number[] }[] = read('../fixtures/sentence-model-parity.json');
const vocab: SentenceVocab = read('../../public/models/sentences-vocab.json');
const model = new URL('../../public/models/sentences-small.onnx', import.meta.url).pathname;

describe('phone sentence model matches Python', () => {
  it.each(cases.map((c, i) => [c.name, c, expected[i]] as const))('%s', async (_name, c, e) => {
    const feats = resampleFrames(c.raws.map(sentenceFrame), c.t);
    const T = feats.length / SF_DIM;
    const session = await ort.InferenceSession.create(model);
    const lp = (await session.run({ feats: new ort.Tensor('float32', feats, [1, T, SF_DIM]) })).logprobs;
    e.logprobs_head.forEach((v, i) => expect((lp.data as Float32Array)[i]).toBeCloseTo(v, 2));
    const { ids } = greedyCtc(lp.data as Float32Array, lp.dims[1], lp.dims[2]);
    expect(ids).toEqual(e.ids);
    expect(glossText(vocab, ids).glosses).toHaveLength(ids.length);
  });
});
