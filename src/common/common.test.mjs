// src/common/common.test.mjs — M0 + 度量单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  cosineSimilarity,
  euclidean,
  recallAtK,
  ndcgAtK,
  precisionRecallF1,
  accuracy,
  nmi,
  jaccard,
} from './metrics.mjs';
import { ErrCode, ForgeError } from './errors.mjs';
import { mulberry32, shuffle } from './rng.mjs';

test('cosineSimilarity: 相同向量=1, 正交=0', () => {
  assert.ok(Math.abs(cosineSimilarity([1, 0], [1, 0]) - 1) < 1e-9);
  assert.ok(Math.abs(cosineSimilarity([1, 0], [0, 1]) - 0) < 1e-9);
});

test('euclidean', () => {
  assert.equal(euclidean([0, 0], [3, 4]), 5);
});

test('recallAtK', () => {
  // 8 个相关，取前 5 命中 3 → 3/8
  assert.equal(recallAtK([1, 2, 3, 4, 5, 6, 7, 8], [3, 9, 1, 10, 2], 5), 3 / 8);
  assert.equal(recallAtK([1], [1], 5), 1);
});

test('ndcgAtK: 理想序 > 乱序', () => {
  const grade = { a: 2, b: 1, c: 0 };
  const ideal = ndcgAtK(['a', 'b', 'c'], grade, 3);
  const bad = ndcgAtK(['c', 'b', 'a'], grade, 3);
  assert.ok(ideal > bad);
  assert.ok(Math.abs(ideal - 1) < 1e-9);
});

test('precisionRecallF1', () => {
  const r = precisionRecallF1([1, 2, 3], [2, 3, 4, 5]);
  assert.equal(r.precision, 2 / 3);
  assert.equal(r.recall, 2 / 4);
  assert.ok(Math.abs(r.f1 - (2 * (2 / 3) * (2 / 4)) / ((2 / 3) + (2 / 4))) < 1e-9);
});

test('accuracy', () => {
  assert.equal(accuracy([1, 0, 1, 0], [1, 0, 0, 0]), 0.75);
});

test('nmi: 完全一致=1', () => {
  assert.ok(Math.abs(nmi([0, 0, 1, 1], [1, 1, 0, 0]) - 1) < 1e-9);
  assert.ok(nmi([0, 0, 1, 1], [0, 1, 0, 1]) < 1); // 弱相关 < 1
});

test('jaccard', () => {
  assert.equal(jaccard([1, 2, 3], [2, 3, 4]), 2 / 4);
});

test('ForgeError 携带 code', () => {
  const e = new ForgeError(ErrCode.VS_EMPTY, '空');
  assert.equal(e.code, 'VS_EMPTY');
  assert.equal(e.toJSON().error, 'VS_EMPTY');
});

test('mulberry32 确定性 + shuffle 不改集合', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  assert.equal(a(), b());
  const s = shuffle([1, 2, 3, 4, 5], mulberry32(7));
  assert.deepEqual(s.slice().sort((x, y) => x - y), [1, 2, 3, 4, 5]);
});
