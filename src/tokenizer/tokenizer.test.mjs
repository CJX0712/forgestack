// src/tokenizer/tokenizer.test.mjs — M1 单测
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tokenizer } from './index.mjs';
import { ErrCode } from '../common/errors.mjs';

const CORPUS =
  '神经网络通过反向传播学习权重，梯度下降最小化损失函数。' +
  'Transformer 用自注意力机制捕捉序列中 token 之间的长距离依赖关系。' +
  '随机森林由多棵决策树组成，通过 bagging 降低方差提升泛化能力。' +
  '过拟合发生在模型记住训练集噪声时，正则化与早停可缓解该问题。' +
  'Embedding 把离散 token 映射到稠密向量空间，相似语义距离更近。' +
  'The quick brown fox jumps over the lazy dog while neural networks learn weights.';

test('train 后 vocab 与 stats 正常', () => {
  const t = new Tokenizer({ merges: 50 }).train(CORPUS);
  assert.ok(t.vocabSize > 0);
  const st = t.stats();
  assert.ok(st.tokens > 0);
  assert.ok(st.ratio > 0 && st.ratio <= 1);
});

test('encode/decode 往返一致（bpe-forge 不变量）', () => {
  const t = new Tokenizer({ merges: 80 }).train(CORPUS);
  const toks = t.encode(CORPUS);
  assert.equal(t.decode(toks), CORPUS);
});

test('不同 merge 数得到不同词表', () => {
  const a = new Tokenizer({ merges: 10 }).train(CORPUS);
  const b = new Tokenizer({ merges: 100 }).train(CORPUS);
  assert.notEqual(a.vocabSize, b.vocabSize);
});

test('空语料抛 TOK_EMPTY', () => {
  const t = new Tokenizer();
  assert.throws(() => t.train(''), (e) => e.code === ErrCode.TOK_EMPTY);
});

test('未训练抛 TOK_UNTRAINED', () => {
  const t = new Tokenizer();
  assert.throws(() => t.encode('x'), (e) => e.code === ErrCode.TOK_UNTRAINED);
});
