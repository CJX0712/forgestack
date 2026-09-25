// src/clustering/index.mjs — M4 Clustering 接口层（复用 spectral-forge 引擎 + k-means 基线）
// Author: 晨星. License: MIT.
import SPEC from './engine.mjs';
import { ErrCode, ForgeError } from '../common/errors.mjs';

export class Clusterer {
  // 主方案：Ng–Jordan–Weiss 谱聚类（非凸结构优于 k-means）
  cluster(vectors, k, opts = {}) {
    if (!Array.isArray(vectors) || vectors.length === 0)
      throw new ForgeError(ErrCode.VS_EMPTY, 'vectors 为空');
    if (!Number.isInteger(k) || k < 2)
      throw new ForgeError(ErrCode.CLU_K_INVALID, 'k 必须为 >=2 的整数', { k });
    if (k > vectors.length)
      throw new ForgeError(ErrCode.CLU_K_INVALID, 'k 不能超过样本数', { k, n: vectors.length });
    const knn = Math.min(opts.knn || 10, vectors.length - 1);
    const res = SPEC.spectralCluster(vectors, {
      k,
      mode: opts.mode || 'knn',
      knn,
      sigma: opts.sigma,
      seed: opts.seed || 42,
    });
    let accuracy = null;
    if (opts.trueLabels) accuracy = SPEC.clusterAccuracy(res.clusters, opts.trueLabels).acc;
    return {
      labels: res.clusters,
      embedding: res.embedding,
      accuracy,
      method: 'spectral',
      laplacianSymmetryError: SPEC.symmetryError(res.M),
      knn,
    };
  }
  // 基线：k-means（复用引擎内置 Lloyd，与谱聚类同宗，便于公平对照）
  baselineKMeans(vectors, k, opts = {}) {
    if (!Number.isInteger(k) || k < 2)
      throw new ForgeError(ErrCode.CLU_K_INVALID, 'k 必须为 >=2 的整数', { k });
    const km = SPEC.kmeans(vectors, k, opts.seed || 42);
    let accuracy = null;
    if (opts.trueLabels) accuracy = SPEC.clusterAccuracy(km.assign, opts.trueLabels).acc;
    return { labels: km.assign, accuracy, method: 'kmeans' };
  }
}
export default Clusterer;
export { SPEC };
