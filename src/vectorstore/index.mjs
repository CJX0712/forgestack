// src/vectorstore/index.mjs — M3 VectorStore 接口层
// 相似度计算复用 OSS：ml-distance（纯 JS similarity.cosine，零原生依赖，可锁版本一键复现）
// 索引为轻量精确扫描：demo 规模（<数千文档）下成本可忽略，且避免 hnswlib/chromadb 的原生构建负担。
// Author: 晨星. License: MIT.
import { similarity } from 'ml-distance';
import { ErrCode, ForgeError } from '../common/errors.mjs';

export class VectorStore {
  constructor(opts = {}) {
    this.items = [];
    this.dim = 0;
    this.metric = opts.metric || 'cosine';
  }
  add(items) {
    if (!Array.isArray(items)) throw new ForgeError(ErrCode.CFG_INVALID, 'add 需要数组');
    for (const it of items) {
      if (!Array.isArray(it.vector)) throw new ForgeError(ErrCode.CFG_INVALID, 'item.vector 必须是数组');
      if (this.dim && it.vector.length !== this.dim)
        throw new ForgeError(ErrCode.VS_DIM_MISMATCH, '向量维度不一致', { expected: this.dim, got: it.vector.length });
      this.dim = it.vector.length;
      this.items.push({ id: it.id, vector: it.vector, meta: it.meta || null });
    }
    return this;
  }
  get size() {
    return this.items.length;
  }
  search(vector, k = 5) {
    if (this.items.length === 0) throw new ForgeError(ErrCode.VS_EMPTY, '索引为空，请先 add()');
    if (vector.length !== this.dim)
      throw new ForgeError(ErrCode.VS_DIM_MISMATCH, '查询向量维度不一致', { expected: this.dim, got: vector.length });
    const scored = this.items.map((it) => {
      const sim = similarity.cosine(vector, it.vector) || 0;
      return { id: it.id, score: sim, meta: it.meta };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }
}
export default VectorStore;
