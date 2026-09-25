// src/eval/eval.test.mjs — M7 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Evaluator } from './index.mjs';

test('recallAtK / ndcgAtK', () => {
  const ev = new Evaluator();
  const rel = [1, 2, 3];
  const retrieved = [2, 9, 1, 10];
  assert.equal(ev.recallAtK(rel, retrieved, 3), 2 / 3);
  assert.ok(ev.ndcgAtK(retrieved, { 1: 1, 2: 1, 3: 1 }, 3) > 0);
});

test('accuracy / nmi', () => {
  const ev = new Evaluator();
  assert.equal(ev.accuracy([1, 0, 1], [1, 0, 1]), 1);
  assert.ok(Math.abs(ev.nmi([0, 0, 1, 1], [1, 1, 0, 0]) - 1) < 1e-9);
});

test('precisionRecallF1', () => {
  const ev = new Evaluator();
  const r = ev.precisionRecallF1([1, 2], [2, 3]);
  assert.equal(r.precision, 0.5);
  assert.equal(r.recall, 0.5);
});
