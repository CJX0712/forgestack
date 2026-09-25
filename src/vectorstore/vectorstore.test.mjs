// src/vectorstore/vectorstore.test.mjs — M3 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VectorStore } from './index.mjs';
import { ErrCode } from '../common/errors.mjs';

test('search 返回最相似在前', () => {
  const vs = new VectorStore();
  vs.add([
    { id: 'a', vector: [1, 0, 0] },
    { id: 'b', vector: [0, 1, 0] },
    { id: 'c', vector: [0, 0, 1] },
  ]);
  const res = vs.search([0.9, 0.1, 0], 2);
  assert.equal(res[0].id, 'a');
  assert.equal(res.length, 2);
});

test('size 累积', () => {
  const vs = new VectorStore();
  vs.add([{ id: 'x', vector: [1, 2] }]);
  assert.equal(vs.size, 1);
});

test('维度不一致抛 VS_DIM_MISMATCH', () => {
  const vs = new VectorStore();
  vs.add([{ id: 'x', vector: [1, 2] }]);
  assert.throws(() => vs.add([{ id: 'y', vector: [1, 2, 3] }]), (e) => e.code === ErrCode.VS_DIM_MISMATCH);
  assert.throws(() => vs.search([1, 2, 3]), (e) => e.code === ErrCode.VS_DIM_MISMATCH);
});

test('空索引抛 VS_EMPTY', () => {
  const vs = new VectorStore();
  assert.throws(() => vs.search([1, 0], 3), (e) => e.code === ErrCode.VS_EMPTY);
});
