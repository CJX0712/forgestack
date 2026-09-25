// src/embeddings/embeddings.test.mjs — M2 单测（tfidf 后端，零依赖确定性）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Embedder } from './index.mjs';
import { ErrCode } from '../common/errors.mjs';

const DOCS = [
  '神经网络通过反向传播学习权重',
  'Transformer 使用自注意力机制捕捉长距离依赖',
  '随机森林通过 bagging 集成多棵决策树',
];

test('fit+embed 确定性 & 维度一致', async () => {
  const e = new Embedder({ backend: 'tfidf', maxTerms: 200 });
  await e.fit(DOCS);
  const a = await e.embed(DOCS[0]);
  const b = await e.embed(DOCS[0]);
  assert.equal(a.backend, 'tfidf');
  assert.ok(a.dims > 0);
  assert.deepEqual(a.vectors[0], b.vectors[0]);
});

test('query 与 docs 在同一向量空间维度', async () => {
  const e = new Embedder({ backend: 'tfidf', maxTerms: 200 });
  await e.fit(DOCS);
  const dv = (await e.embed(DOCS)).vectors;
  const qv = (await e.embed('注意力机制在 Transformer 中的应用')).vectors;
  assert.equal(dv[0].length, qv[0].length);
});

test('空语料抛 EMB_EMPTY', async () => {
  const e = new Embedder();
  await assert.rejects(() => e.fit([]), (err) => err.code === ErrCode.EMB_EMPTY);
});

test('未 fit 抛 EMB_LOAD', async () => {
  const e = new Embedder();
  await assert.rejects(() => e.embed('x'), (err) => err.code === ErrCode.EMB_LOAD);
});

test('未知 backend 抛 CFG_INVALID', async () => {
  const e = new Embedder({ backend: 'nope' });
  await assert.rejects(() => e.fit(DOCS), (err) => err.code === ErrCode.CFG_INVALID);
});
