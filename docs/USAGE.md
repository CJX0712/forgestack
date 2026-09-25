# 使用指南（Usage）

> 作者：晨星（CJX0712） · License：MIT
> 每个模块都可独立使用：先 `import`，再调用其接口；下面给出最小可运行示例。

## 1. M1 Tokenizer（BPE 分词）

```js
import { Tokenizer } from './src/tokenizer/index.mjs';

const tok = new Tokenizer({ merges: 120 });
tok.train('神经网络通过反向传播学习权重。transformer 用注意力机制。');
const toks = tok.encode('注意力机制决定了推理成本');
console.log(toks);            // string[]（BPE 子词序列）
console.log(tok.decode(toks)); // 还原原文
console.log(tok.stats());      // {chars,bytes,tokens,ratio,bytesPerToken,...}
```

## 2. M2 Embeddings（向量化）

```js
import { Embedder } from './src/embeddings/index.mjs';

const emb = new Embedder({ backend: 'tfidf', maxTerms: 1500 });
await emb.fit(['文档一', '文档二']);
const { vectors, backend, dims } = await emb.embed(['查询语句']);
// vectors[0] 为查询向量，与 docs 同维度
```

> 切换稠密后端：`new Embedder({ backend: 'minilm' })`（需 `npm i @xenova/transformers`）。

## 3. M3 VectorStore（向量检索）

```js
import { VectorStore } from './src/vectorstore/index.mjs';

const vs = new VectorStore();
vs.add([
  { id: 'd1', vector: [1, 0, 0] },
  { id: 'd2', vector: [0, 1, 0] },
]);
const top = vs.search([0.9, 0.1, 0], 1); // [{ id:'d1', score, meta }]
```

## 4. M4 Clustering（谱聚类）

```js
import { Clusterer } from './src/clustering/index.mjs';

const c = new Clusterer();
const res = c.cluster(vectors, 3, { knn: 8, seed: 42, trueLabels });
console.log(res.labels, res.accuracy, res.laplacianSymmetryError);
// 基线对照
const km = c.baselineKMeans(vectors, 3, { seed: 42, trueLabels });
```

## 5. M5 Ranker（随机森林重排器）

```js
import { Ranker } from './src/ranker/index.mjs';

const r = new Ranker({ nTrees: 64, seed: 7 });
r.fit(featureMatrix, labels);             // featureMatrix: number[][], labels: number[]
const probs = r.predictProb(candidatesFeatures);
// 或直接重排候选
const ranked = r.rerank(candidates, (c) => featureOf(c));
```

## 6. M7 Eval（评测）

```js
import { Evaluator } from './src/eval/index.mjs';

const ev = new Evaluator();
ev.recallAtK(relevantIds, retrievedIds, 5);
ev.ndcgAtK(retrievedIds, gradeMap, 5);
ev.nmi(trueLabels, predLabels);
```

## 7. M6 Orchestrator（端到端）

```js
import { Pipeline } from './src/orchestrator/index.mjs';
import { DOCS, QUERIES } from './src/orchestrator/corpus.mjs';

const pipe = new Pipeline({ backend: 'tfidf', k: 3, seed: 42 });
await pipe.ingest(DOCS);
const res = await pipe.query(QUERIES[0].q, 5);
console.log(res.answer, res.reranked);
console.log('topics:', pipe.topics());
console.log('bench:', pipe.benchmarkClustering());
```

## 8. 替换/扩展模块

- **换嵌入后端**：在 `M2` 增加 `backend` 分支（如 `openai` / `bge`）实现 `fit/embed` 同一接口即可，其余模块无需改动。
- **换向量索引**：`M3` 替换为 ANN 索引（hnswlib），只要保留 `add(items)` 与 `search(vector,k)` 签名。
- **换聚类/重排算法**：`M4`/`M5` 替换为其它引擎，保持返回结构（`labels` / `{predictProbOne}`）即可被 `M6` 直接复用。
