// src/clustering/clustering.test.mjs — M4 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Clusterer } from './index.mjs';
import SPEC from './engine.mjs';
import { ErrCode } from '../common/errors.mjs';

test('谱聚类在 moons 上显著优于 k-means', () => {
  const moons = SPEC.makeMoons(40, 40, 2);
  const c = new Clusterer();
  const sp = c.cluster(moons.X, 2, { knn: 10, seed: 42, trueLabels: moons.y });
  const km = c.baselineKMeans(moons.X, 2, { seed: 42, trueLabels: moons.y });
  assert.ok(sp.accuracy > 0.9, `spectral acc=${sp.accuracy}`);
  assert.ok(sp.accuracy > km.accuracy, `spectral ${sp.accuracy} > kmeans ${km.accuracy}`);
});

test('谱聚类在 circles 上拉满、k-means 失败', () => {
  const circ = SPEC.makeCircles(40, 40, 2);
  const c = new Clusterer();
  const sp = c.cluster(circ.X, 2, { knn: 10, seed: 42, trueLabels: circ.y });
  const km = c.baselineKMeans(circ.X, 2, { seed: 42, trueLabels: circ.y });
  assert.ok(sp.accuracy > 0.9);
  assert.ok(km.accuracy < 0.7);
});

test('拉普拉斯矩阵对称误差极小', () => {
  const moons = SPEC.makeMoons(30, 30, 2);
  const c = new Clusterer();
  const sp = c.cluster(moons.X, 2, { knn: 8, seed: 1 });
  assert.ok(sp.laplacianSymmetryError < 1e-6);
});

test('k<2 抛 CLU_K_INVALID', () => {
  const c = new Clusterer();
  const moons = SPEC.makeMoons(20, 20, 2);
  assert.throws(() => c.cluster(moons.X, 1), (e) => e.code === ErrCode.CLU_K_INVALID);
});
