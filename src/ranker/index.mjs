// src/ranker/index.mjs — M5 Ranker 接口层（复用 forest-forge 随机森林作学习型重排器）
// Author: 晨星. License: MIT.
import FOREST from './engine.mjs';
import { ErrCode, ForgeError } from '../common/errors.mjs';

export class Ranker {
  constructor(opts = {}) {
    this.opts = {
      nTrees: opts.nTrees ?? 64,
      maxDepth: opts.maxDepth ?? 12,
      minSamplesLeaf: opts.minSamplesLeaf ?? 1,
      mtry: opts.mtry ?? 'sqrt',
      seed: opts.seed ?? 42,
    };
    this.rf = null;
  }
  fit(X, y) {
    if (!Array.isArray(X) || X.length < 2)
      throw new ForgeError(ErrCode.RNK_INSUFFICIENT, '训练样本不足', { n: X ? X.length : 0 });
    const nFeat = X[0].length;
    for (const row of X) {
      if (!Array.isArray(row) || row.length !== nFeat)
        throw new ForgeError(ErrCode.CFG_INVALID, '特征维度不一致', { expected: nFeat, got: row ? row.length : -1 });
    }
    this.rf = new FOREST.RandomForest(this.opts).fit(X, y);
    this.nFeat = nFeat;
    return this;
  }
  predict(X) {
    if (!this.rf) throw new ForgeError(ErrCode.RNK_UNFIT, '模型未训练，请先 fit()');
    return this.rf.predict(X);
  }
  predictProb(X) {
    if (!this.rf) throw new ForgeError(ErrCode.RNK_UNFIT, '模型未训练，请先 fit()');
    return X.map((x) => this.rf.predictProbOne(x));
  }
  get importances() {
    if (!this.rf) throw new ForgeError(ErrCode.RNK_UNFIT, '模型未训练');
    return this.rf.importances;
  }
  // 给定候选集与特征函数，返回按相关性概率降序排列的候选列表
  rerank(candidates, featurize, opts = {}) {
    if (!this.rf) throw new ForgeError(ErrCode.RNK_UNFIT, '模型未训练');
    const scored = candidates.map((c) => {
      const f = featurize(c);
      const p = this.rf.predictProbOne(f);
      const rel = p[1] !== undefined ? p[1] : Math.max(...Object.values(p));
      return { ...c, _score: rel };
    });
    scored.sort((a, b) => b._score - a._score);
    return scored;
  }
}
export default Ranker;
export { FOREST };
