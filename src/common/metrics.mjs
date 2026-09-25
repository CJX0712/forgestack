// src/common/metrics.mjs — 标准评估度量（纯 JS 实现）
// 说明：本环境配置的 npm registry（ohpm.openharmony.cn）无法解析经典评测包 ml-metrics，
// 且公共 npm 上 ml-metrics 已不可得。为满足"干净环境一键复现"的 DoD，这里按标准定义
// 自行实现，语义与 scikit-learn 对齐（见各函数注释）。如需切换为 OSS 实现，把对应函数
// 替换为 ml-metrics / ml-distance 的调用即可，接口不变。
// Author: 晨星. License: MIT.

// 余弦相似度（兜底/测试用；生产路径 VectorStore 复用 OSS ml-distance）
export function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

// 安全余弦：任一侧为零向量时返回 0（避免 0/0=NaN；用于查询/文档向量稀疏场景）
export function safeCosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const da = Math.sqrt(na);
  const db = Math.sqrt(nb);
  if (da === 0 || db === 0) return 0;
  return dot / (da * db);
}

// recall@k：检索到的前 k 个中命中相关文档的比例（sklearn 无直接等价，IR 标准定义）
export function recallAtK(relevantIds, retrievedIds, k) {
  const r = new Set(relevantIds);
  let hit = 0;
  const n = Math.min(k, retrievedIds.length);
  for (let i = 0; i < n; i++) if (r.has(retrievedIds[i])) hit++;
  return r.size === 0 ? 0 : hit / r.size;
}

// NDCG@k：带分级相关性的归一化折损累计增益（sklearn 无；与 pyltr / 搜索引擎标准一致）
function dcg(grades) {
  let d = 0;
  for (let i = 0; i < grades.length; i++) {
    d += (Math.pow(2, grades[i]) - 1) / Math.log2(i + 2);
  }
  return d;
}
export function ndcgAtK(retrievedIds, gradeMap, k) {
  const top = retrievedIds.slice(0, k).map((id) => gradeMap[id] || 0);
  const ideal = Object.values(gradeMap).slice().sort((a, b) => b - a).slice(0, k);
  const idcg = dcg(ideal);
  return idcg === 0 ? 0 : dcg(top) / idcg;
}

// 集合精确率/召回率/F1（sklearn.metrics.precision/recall/f1_score 二值集合语义）
export function precisionRecallF1(predictedSet, relevantSet) {
  const r = new Set(relevantSet);
  const p = new Set(predictedSet);
  let hit = 0;
  p.forEach((x) => { if (r.has(x)) hit++; });
  const prec = p.size === 0 ? 0 : hit / p.size;
  const rec = r.size === 0 ? 0 : hit / r.size;
  const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
  return { precision: prec, recall: rec, f1 };
}

// 分类准确率（sklearn.metrics.accuracy_score）
export function accuracy(predicted, truth) {
  if (predicted.length !== truth.length) throw new Error('length mismatch');
  let hit = 0;
  for (let i = 0; i < predicted.length; i++) if (predicted[i] === truth[i]) hit++;
  return hit / predicted.length;
}

// 归一化互信息 NMI（sklearn.metrics.normalized_mutual_info_score，算术平均归一化）
export function nmi(trueLabels, predLabels) {
  const N = trueLabels.length;
  if (N === 0) return 1;
  const cMap = [...new Set(trueLabels)];
  const kMap = [...new Set(predLabels)];
  const cIdx = new Map(cMap.map((c, i) => [c, i]));
  const kIdx = new Map(kMap.map((c, i) => [c, i]));
  const C = cMap.length, K = kMap.length;
  const M = Array.from({ length: C }, () => new Array(K).fill(0));
  for (let i = 0; i < N; i++) M[cIdx.get(trueLabels[i])][kIdx.get(predLabels[i])]++;
  const aSum = new Array(C).fill(0);
  const bSum = new Array(K).fill(0);
  for (let i = 0; i < C; i++) for (let j = 0; j < K; j++) { aSum[i] += M[i][j]; bSum[j] += M[i][j]; }
  let I = 0;
  for (let i = 0; i < C; i++) for (let j = 0; j < K; j++) {
    const n = M[i][j];
    if (n > 0) I += (n / N) * (Math.log2((n * N) / ((aSum[i] * bSum[j]) || 1)));
  }
  let Ht = 0, Hp = 0;
  for (let i = 0; i < C; i++) { const p = aSum[i] / N; if (p > 0) Ht -= p * Math.log2(p); }
  for (let j = 0; j < K; j++) { const p = bSum[j] / N; if (p > 0) Hp -= p * Math.log2(p); }
  const denom = (Ht + Hp) / 2;
  return denom === 0 ? 1 : I / denom;
}

export function jaccard(aSet, bSet) {
  const a = new Set(aSet), b = new Set(bSet);
  let inter = 0;
  a.forEach((x) => { if (b.has(x)) inter++; });
  const uni = new Set([...a, ...b]).size;
  return uni === 0 ? 0 : inter / uni;
}
