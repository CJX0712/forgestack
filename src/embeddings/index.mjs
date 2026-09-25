// src/embeddings/index.mjs — M2 Embeddings 接口层
// 后端策略：
//   tfidf  —— 纯 JS、零依赖、确定性（默认，保证干净环境一键跑通）
//   minilm —— 复用业界 SOTA 句向量模型 all-MiniLM-L6-v2（transformers.js / @xenova/transformers，
//             需用户自行 `npm i transformers.js` 后启用；非默认依赖，避免原生构建破坏可复现性）
// Author: 晨星. License: MIT.
import { ErrCode, ForgeError } from '../common/errors.mjs';
import { logger } from '../common/logger.mjs';

const CJK = /[一-鿿]/;
function tokenize(text) {
  const toks = [];
  const re = /[a-zA-Z0-9]+|[一-鿿]/g;
  let m;
  while ((m = re.exec(text)) !== null) toks.push(m[0].toLowerCase());
  return toks;
}

export class Embedder {
  constructor(opts = {}) {
    this.backend = opts.backend || 'tfidf';
    this.maxTerms = opts.maxTerms || 2000;
    this.vocab = null;
    this.idf = null;
    this.dim = 0;
    this._model = null;
  }
  async fit(corpusTexts) {
    if (!Array.isArray(corpusTexts) || corpusTexts.length === 0)
      throw new ForgeError(ErrCode.EMB_EMPTY, 'fit 语料为空');
    if (this.backend === 'tfidf') {
      this._fitTfidf(corpusTexts);
      return this;
    }
    if (this.backend === 'minilm') {
      await this._loadMiniLM();
      return this;
    }
    throw new ForgeError(ErrCode.CFG_INVALID, '未知 embedding backend', { backend: this.backend });
  }
  async embed(texts) {
    const arr = Array.isArray(texts) ? texts : [texts];
    if (this.backend === 'tfidf') {
      if (!this.vocab) throw new ForgeError(ErrCode.EMB_LOAD, 'tfidf 尚未 fit');
      const vecs = arr.map((t) => this._tfidfVec(t));
      return { vectors: vecs, backend: 'tfidf', dims: this.dim };
    }
    await this._loadMiniLM();
    const out = await this._model(arr, { pooling: 'mean', normalize: true });
    const vecs = out.tolist();
    return { vectors: vecs, backend: 'minilm', dims: vecs[0].length };
  }
  _fitTfidf(corpus) {
    const N = corpus.length;
    const df = new Map();
    const docToks = corpus.map((doc) => {
      const toks = tokenize(doc);
      const seen = new Set();
      for (const t of toks) seen.add(t);
      for (const t of seen) df.set(t, (df.get(t) || 0) + 1);
      return toks;
    });
    // 取 df 最高的 maxTerms 个词项作为词表
    const ranked = [...df.entries()].sort((a, b) => b[1] - a[1]).slice(0, this.maxTerms);
    const vocab = new Map();
    ranked.forEach(([term], i) => vocab.set(term, i));
    const idf = new Map();
    for (const [term, d] of ranked) idf.set(term, Math.log(N / d) + 1);
    this.vocab = vocab;
    this.idf = idf;
    this.dim = vocab.size;
    this._docToks = docToks;
    logger.info('tfidf 拟合完成', { docs: N, vocab: this.dim });
  }
  _tfidfVec(text) {
    const toks = tokenize(text);
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    const v = new Array(this.dim).fill(0);
    let norm = 0;
    for (const [term, c] of tf) {
      const idx = this.vocab.get(term);
      if (idx === undefined) continue;
      const w = c * this.idf.get(term);
      v[idx] = w;
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
  }
  async _loadMiniLM() {
    if (this._model) return;
    let mod = null;
    const candidates = ['transformers.js', '@xenova/transformers'];
    for (const name of candidates) {
      try {
        mod = await import(name);
        break;
      } catch (e) {
        logger.debug(`MiniLM 后端不可用: ${name}`, { err: e.message });
      }
    }
    if (!mod) throw new ForgeError(ErrCode.EMB_LOAD, '未安装 transformers 后端，请执行 npm i transformers.js', { tried: candidates });
    const pipeline = mod.pipeline || (mod.default && mod.default.pipeline);
    if (!pipeline) throw new ForgeError(ErrCode.EMB_LOAD, 'transformers 模块缺少 pipeline API');
    this._model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    logger.info('MiniLM 后端已加载', { model: 'Xenova/all-MiniLM-L6-v2' });
  }
}
export default Embedder;
