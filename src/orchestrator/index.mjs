// src/orchestrator/index.mjs — M6 Orchestrator（编排全链路）
// 主链路：ingest → Tokenizer(分词/特征) → Embeddings(向量) → VectorStore(检索)
//                                                    ↘ Clusterer(谱聚类主题发现)
//         query → VectorStore 检索 → Ranker(随机森林重排) → answer
// 复用：bpe-forge / spectral-forge / forest-forge 引擎 + ml-distance 余弦 + transformers.js(可选 SOTA)
// Author: 晨星. License: MIT.
import { Tokenizer } from '../tokenizer/index.mjs';
import { Embedder } from '../embeddings/index.mjs';
import { VectorStore } from '../vectorstore/index.mjs';
import { Clusterer } from '../clustering/index.mjs';
import { Ranker } from '../ranker/index.mjs';
import { Evaluator } from '../eval/index.mjs';
import SPEC from '../clustering/engine.mjs';
import { safeCosine } from '../common/metrics.mjs';
import { logger } from '../common/logger.mjs';
import { ErrCode, ForgeError } from '../common/errors.mjs';

function encodeTopics(truth) {
  const uniq = [...new Set(truth)];
  const m = new Map(uniq.map((t, i) => [t, i]));
  return truth.map((t) => m.get(t));
}

export class Pipeline {
  constructor(opts = {}) {
    this.backend = opts.backend || 'tfidf';
    this.k = opts.k || 3;
    this.retrieveK = opts.retrieveK || 10;
    this.knn = opts.knn || 8;
    this.seed = opts.seed || 42;
    this.tokenizer = new Tokenizer({ merges: opts.merges ?? 200 });
    this.embedder = new Embedder({ backend: this.backend, maxTerms: opts.maxTerms ?? 1500 });
    this.store = new VectorStore();
    this.clusterer = new Clusterer();
    this.evaluator = new Evaluator();
    this.ranker = new Ranker({ seed: this.seed });
    this.docs = null;
    this.docVecs = null;
    this.docTok = null;
    this.clusterLabels = null;
    this.clusterCentroids = null;
    this.embedBackend = null;
    this.embedDims = 0;
  }

  async ingest(docs) {
    if (!Array.isArray(docs) || docs.length === 0)
      throw new ForgeError(ErrCode.ORC_NOT_INGESTED, 'docs 为空');
    this.docs = docs;
    const corpus = docs.map((d) => d.text).join('\n');
    this.tokenizer.train(corpus, this.tokenizer.merges);
    this.docTok = docs.map((d) => new Set(this.tokenizer.encode(d.text)));

    const texts = docs.map((d) => d.text);
    await this.embedder.fit(texts);
    const { vectors, backend, dims } = await this.embedder.embed(texts);
    this.docVecs = vectors;
    this.embedBackend = backend;
    this.embedDims = dims;

    this.store.add(docs.map((d, i) => ({ id: d.id, vector: vectors[i], meta: { topic: d.topic, text: d.text } })));

    const knn = Math.min(this.knn, docs.length - 1);
    const res = this.clusterer.cluster(vectors, this.k, { knn, seed: this.seed });
    this.clusterLabels = res.labels;
    this.clusterMethod = res.method;
    this.clusterCentroids = this._centroids(vectors, res.labels, this.k);
    this._trainRanker();
    logger.info('ingest 完成', { docs: docs.length, backend, dims, clusters: this.k });
    return this;
  }

  _centroids(vectors, labels, k) {
    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const cnt = new Array(k).fill(0);
    labels.forEach((l, i) => {
      if (l < 0 || l >= k) return;
      cnt[l]++;
      vectors[i].forEach((v, j) => (sums[l][j] += v));
    });
    return sums.map((s, c) => (cnt[c] ? s.map((v) => v / cnt[c]) : s));
  }

  _queryCluster(qVec) {
    let best = -1;
    let bestS = -Infinity;
    for (let c = 0; c < this.clusterCentroids.length; c++) {
      const s = safeCosine(qVec, this.clusterCentroids[c]);
      if (s > bestS) {
        bestS = s;
        best = c;
      }
    }
    return best;
  }

  _features(qVec, qTok, candIdx) {
    const dVec = this.docVecs[candIdx];
    const dTok = this.docTok[candIdx];
    const sim = safeCosine(qVec, dVec);
    let inter = 0;
    qTok.forEach((t) => {
      if (dTok.has(t)) inter++;
    });
    const uni = new Set([...qTok, ...dTok]).size;
    const lex = uni ? inter / uni : 0;
    const qc = this._queryCluster(qVec);
    const sameCluster = qc === this.clusterLabels[candIdx] ? 1 : 0;
    const lr = qTok.size && dTok.size ? Math.min(qTok.size, dTok.size) / Math.max(qTok.size, dTok.size) : 0;
    return [sim, lex, sameCluster, lr];
  }

  _trainRanker() {
    const X = [];
    const y = [];
    const n = this.docs.length;
    for (let i = 0; i < n; i++) {
      const qTok = this.docTok[i];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const f = this._features(this.docVecs[i], qTok, j);
        X.push(f);
        y.push(this.docs[i].topic === this.docs[j].topic ? 1 : 0);
    }
    }
    this.ranker.fit(X, y);
    logger.info('ranker 训练完成', { pairs: X.length, importances: this.ranker.importances.map((v) => v.toFixed(3)) });
  }

  async query(q, k = 5) {
    if (!this.docs) throw new ForgeError(ErrCode.ORC_NOT_INGESTED, '尚未 ingest');
    if (!q || !q.trim()) throw new ForgeError(ErrCode.ORC_EMPTY_QUERY, 'query 为空');
    const { vectors: [qVec] } = await this.embedder.embed([q]);
    const qTok = new Set(this.tokenizer.encode(q));
    const retrieved = this.store.search(qVec, Math.min(this.retrieveK, this.docs.length));
    const idxOf = new Map(this.docs.map((d, i) => [d.id, i]));
    const ranked = this.ranker.rerank(
      retrieved.map((r) => ({ id: r.id, idx: idxOf.get(r.id), score0: r.score })),
      (c) => this._features(qVec, qTok, c.idx)
    );
    return {
      query: q,
      queryCluster: this._queryCluster(qVec),
      retrieved: retrieved.map((r) => ({ id: r.id, score: r.score, topic: r.meta.topic })),
      reranked: ranked.map((c) => ({ id: c.id, score: c._score, topic: this.docs[c.idx].topic })),
      answer: this.docs[ranked[0].idx].text,
    };
  }

  // 主题发现：谱聚类 vs k-means 基线（在文档嵌入空间，用 ground-truth topic 评估）
  topics() {
    if (!this.docs) throw new ForgeError(ErrCode.ORC_NOT_INGESTED, '尚未 ingest');
    const truth = encodeTopics(this.docs.map((d) => d.topic));
    const knn = Math.min(this.knn, this.docs.length - 1);
    const spectral = this.clusterer.cluster(this.docVecs, this.k, { knn, seed: this.seed, trueLabels: truth });
    const km = this.clusterer.baselineKMeans(this.docVecs, this.k, { seed: this.seed, trueLabels: truth });
    return { spectralAcc: spectral.accuracy, kmeansAcc: km.accuracy, laplacianSymErr: spectral.laplacianSymmetryError };
  }

  // 合成非凸基准：谱聚类应在 moons / circles 上显著优于 k-means（NJW 经典结论）
  benchmarkClustering() {
    const moons = SPEC.makeMoons(40, 40, 2);
    const sp = this.clusterer.cluster(moons.X, 2, { knn: 10, seed: 42, trueLabels: moons.y });
    const km = this.clusterer.baselineKMeans(moons.X, 2, { seed: 42, trueLabels: moons.y });
    const circles = SPEC.makeCircles(40, 40, 2);
    const spc = this.clusterer.cluster(circles.X, 2, { knn: 10, seed: 42, trueLabels: circles.y });
    const kmc = this.clusterer.baselineKMeans(circles.X, 2, { seed: 42, trueLabels: circles.y });
    return {
      moons: { spectral: sp.accuracy, kmeans: km.accuracy },
      circles: { spectral: spc.accuracy, kmeans: kmc.accuracy },
    };
  }

  get stats() {
    return {
      backend: this.embedBackend,
      dims: this.embedDims,
      nDocs: this.docs ? this.docs.length : 0,
      clusters: this.k,
      rankerImportances: this.ranker.importances ? this.ranker.importances.map((v) => v.toFixed(3)) : null,
    };
  }
}
export default Pipeline;
