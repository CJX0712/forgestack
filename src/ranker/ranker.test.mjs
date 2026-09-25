// src/ranker/ranker.test.mjs — M5 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Ranker } from './index.mjs';
import FOREST from './engine.mjs';
import { ErrCode } from '../common/errors.mjs';

test('在可分 blobs 上 fit/predict 高准确率', () => {
  const d = FOREST.makeDataset('blobs', 120, 7, 0.04);
  const r = new Ranker({ seed: 7 }).fit(d.X, d.y);
  const pred = r.predict(d.X);
  let hit = 0;
  for (let i = 0; i < d.X.length; i++) if (pred[i] === d.y[i]) hit++;
  assert.ok(hit / d.X.length > 0.85, `acc=${(hit / d.X.length).toFixed(3)}`);
});

test('特征重要性归一化和≈1', () => {
  const d = FOREST.makeDataset('blobs', 100, 3, 0.03);
  const r = new Ranker({ seed: 3 }).fit(d.X, d.y);
  const sum = r.importances.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);
});

test('rerank 按相关性概率降序', () => {
  const d = FOREST.makeDataset('blobs', 80, 5, 0.03);
  const r = new Ranker({ seed: 5 }).fit(d.X, d.y);
  const cands = d.X.slice(0, 6).map((x, i) => ({ id: 'c' + i, idx: i }));
  const ranked = r.rerank(cands, (c) => d.X[c.idx]);
  for (let i = 1; i < ranked.length; i++) assert.ok(ranked[i - 1]._score >= ranked[i]._score);
});

test('样本不足抛 RNK_INSUFFICIENT', () => {
  const r = new Ranker();
  assert.throws(() => r.fit([[0]], [0]), (e) => e.code === ErrCode.RNK_INSUFFICIENT);
});

test('未训练抛 RNK_UNFIT', () => {
  const r = new Ranker();
  assert.throws(() => r.predict([[0, 0]]), (e) => e.code === ErrCode.RNK_UNFIT);
});
