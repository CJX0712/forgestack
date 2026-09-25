// src/orchestrator/orchestrator.test.mjs — M6 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Pipeline } from './index.mjs';
import { DOCS, QUERIES } from './corpus.mjs';
import { ErrCode } from '../common/errors.mjs';

test('ingest + query 全链路', async () => {
  const p = new Pipeline({ seed: 1 });
  await p.ingest(DOCS);
  const res = await p.query(QUERIES[0].q, 5);
  assert.equal(res.query, QUERIES[0].q);
  assert.ok(typeof res.answer === 'string' && res.answer.length > 0);
  assert.equal(res.reranked.length, res.retrieved.length);
  assert.ok(res.reranked.length > 0);
  assert.ok(res.reranked[0].score >= res.reranked[res.reranked.length - 1].score);
});

test('topics: 谱聚类 accuracy 可计算', () => {
  const p = new Pipeline({ seed: 1 });
  return p.ingest(DOCS).then(() => {
    const t = p.topics();
    assert.ok(t.spectralAcc >= 0 && t.spectralAcc <= 1);
    assert.ok(t.kmeansAcc >= 0 && t.kmeansAcc <= 1);
  });
});

test('benchmarkClustering: 谱聚类在 moons/circles 优于 kmeans', () => {
  const p = new Pipeline({ seed: 1 });
  return p.ingest(DOCS).then(() => {
    const b = p.benchmarkClustering();
    assert.ok(b.moons.spectral > b.moons.kmeans);
    assert.ok(b.circles.spectral > b.circles.kmeans);
  });
});

test('空 docs 抛 ORC_NOT_INGESTED', async () => {
  const p = new Pipeline();
  await assert.rejects(() => p.ingest([]), (e) => e.code === ErrCode.ORC_NOT_INGESTED);
});

test('未 ingest 查询抛错', async () => {
  const p = new Pipeline();
  await assert.rejects(() => p.query('x'), (e) => e.code === ErrCode.ORC_NOT_INGESTED);
});
