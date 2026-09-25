# ForgeStack 系统架构（Architecture）

> 作者：晨星（CJX0712） · License：MIT
> 本文档定义模块边界、接口契约（输入/输出/协议/错误码）、调用关系与选型依据。

## 1. 架构图（模块与调用关系）

```
                 ┌──────────── 输入: docs[] / query ────────────┐
                 ▼                                             │
   [M1 Tokenizer]──(corpus, token预算)──▶ [M2 Embeddings]──(texts)──▶ vectors
   (bpe-forge)                                │ tfidf | minilm(SOTA)        │
                                              ▼                           │
                                   [M3 VectorStore]──(add ids,vectors)   │
                                   (ml-distance cosine)                   │
                                              │ search(vec,k)              │
                                              ▼                           │
                                   [M5 Ranker]──(rerank candidates)      │
                                   (forest-forge RF)                     │
                                              │                           │
   [M4 Clustering]──(doc vectors, k)──▶ 主题发现(非凸,谱聚类 > k-means)   │
   (spectral-forge)                           │                           │
                                              ▼                           │
                                   [M6 Orchestrator]──▶ answer + debug    │
                                              │                           │
                                   [M7 Eval]──(recall@k / NDCG / acc) ◀──┘
                 M0 Common: 错误码枚举 / 日志 / RNG / 标准度量
```

调用关系：M6 编排 M1→M2→M3→M5 主链路；M4 离线主题分析；M7 横切评测；M0 提供统一错误码与日志。

## 2. 模块接口契约

### M0 Common（`src/common`）
| 组件 | 接口 | 说明 |
|------|------|------|
| errors | `ForgeError(code, msg, details)` / `ErrCode.*` | 统一错误码与错误对象 |
| logger | `logger.info/warn/error(msg, meta)` / `setLevel(name)` | 分级日志 |
| rng | `mulberry32(seed)` / `shuffle(arr, rng)` / `splitIndices(n, frac, rng)` | 可复现随机 |
| metrics | `cosineSimilarity / safeCosine / recallAtK / ndcgAtK / precisionRecallF1 / accuracy / nmi / jaccard` | 标准度量（sklearn 等价语义） |

### M1 Tokenizer（`src/tokenizer`，复用 bpe-forge）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `train(corpus, merges?)` | string, int | `this`（含 `model`） | `TOK_EMPTY`, `TOK_BAD_MERGE` |
| `encode(text)` | string | `string[]`（BPE token 序列） | `TOK_UNTRAINED` |
| `decode(tokens)` | `string[]` | string | — |
| `stats()` | — | `{chars,bytes,tokens,ratio,bytesPerToken,uniqTokens}` | `TOK_UNTRAINED` |
| `vocabSize` / `tokenFreq()` | — | number / `[[token,count]]` | `TOK_UNTRAINED` |

### M2 Embeddings（`src/embeddings`，复用 transformers.js / 自研 tfidf）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `fit(corpusTexts[])` | string[] | `this` | `EMB_EMPTY`, `CFG_INVALID` |
| `embed(texts)` | string \| string[] | `{vectors:number[][], backend, dims}` | `EMB_LOAD` |
| 后端 | `tfidf`（默认，确定/离线）\| `minilm`（需 `npm i @xenova/transformers`） | — | `EMB_LOAD`（模型不可用） |

### M3 VectorStore（`src/vectorstore`，复用 ml-distance）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `add(items:[{id,vector,meta?}])` | — | `this` | `VS_DIM_MISMATCH`, `CFG_INVALID` |
| `search(vector, k)` | number[], int | `[{id,score,meta}]`（降序） | `VS_EMPTY`, `VS_DIM_MISMATCH` |
| `size` | — | number | — |

### M4 Clustering（`src/clustering`，复用 spectral-forge）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `cluster(vectors, k, {mode,knn,sigma,seed,trueLabels?})` | number[][], int | `{labels,embedding,accuracy?,laplacianSymmetryError,method:'spectral'}` | `CLU_K_INVALID`, `VS_EMPTY` |
| `baselineKMeans(vectors, k, {seed,trueLabels?})` | number[][], int | `{labels,accuracy?,method:'kmeans'}` | `CLU_K_INVALID` |

### M5 Ranker（`src/ranker`，复用 forest-forge）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `fit(X:number[][], y:number[])` | 特征矩阵, 标签 | `this` | `RNK_INSUFFICIENT`, `CFG_INVALID` |
| `predict(X)` / `predictProb(X)` | number[][] | 标签[] / 概率对象[] | `RNK_UNFIT` |
| `importances` | — | number[]（归一化和≈1） | `RNK_UNFIT` |
| `rerank(candidates, featurize)` | 候选[], 特征函数 | 按相关性概率降序的候选[] | `RNK_UNFIT` |

### M6 Orchestrator（`src/orchestrator`）
| 方法 | 输入 | 输出 | 错误码 |
|------|------|------|--------|
| `ingest(docs:[{id,text,topic?}])` | — | `this`（训练 tokenizer/embedder/store/clusterer/ranker） | `ORC_NOT_INGESTED` |
| `query(q, k)` | string, int | `{query,queryCluster,retrieved,reranked,answer}` | `ORC_NOT_INGESTED`, `ORC_EMPTY_QUERY` |
| `topics()` | — | `{spectralAcc,kmeansAcc,laplacianSymErr}` | `ORC_NOT_INGESTED` |
| `benchmarkClustering()` | — | `{moons:{spectral,kmeans},circles:{...}}` | — |

### M7 Eval（`src/eval`）
| 方法 | 输入 | 输出 |
|------|------|------|
| `recallAtK(relIds, retrievedIds, k)` | id 集合/列表, int | number |
| `ndcgAtK(retrievedIds, gradeMap, k)` | 列表, {id:grade}, int | number |
| `precisionRecallF1(predSet, relSet)` | Set[] | `{precision,recall,f1}` |
| `accuracy(predicted[], truth[])` | number[] | number |
| `nmi(trueLabels[], predLabels[])` | number[] | number |

## 3. 错误码登记表

| 错误码 | 含义 | 触发模块 |
|--------|------|----------|
| `TOK_EMPTY` / `TOK_BAD_MERGE` / `TOK_UNTRAINED` | 分词器空语料 / merge 非法 / 未训练 | M1 |
| `EMB_EMPTY` / `EMB_LOAD` | 嵌入语料空 / 模型或后端不可用 | M2 |
| `VS_DIM_MISMATCH` / `VS_EMPTY` | 向量维度不一致 / 索引空 | M3 |
| `CLU_K_INVALID` / `CLU_SINGULAR` | k 非法 | M4 |
| `RNK_UNFIT` / `RNK_INSUFFICIENT` | 未训练 / 样本不足 | M5 |
| `EVAL_NO_GT` | 缺 ground truth | M7 |
| `ORC_NOT_INGESTED` / `ORC_EMPTY_QUERY` | 未摄取 / 空查询 | M6 |
| `CFG_INVALID` | 配置非法 | 全局 |

## 4. 选型依据（性能 / 生态 / 许可证 / 活跃度）

| 选型 | 候选对比 | 决定 | 许可证 | 理由 |
|------|----------|------|--------|------|
| 嵌入 | transformers.js(MiniLM) vs 自研 vs fastembed | **transformers.js**（可选启用）+ tfidf 兜底 | MIT/Apache | SOTA 句向量；Node WASM 零原生编译；tfidf 兜底保证 DoD |
| 距离 | ml-distance vs hnswlib-node vs chromadb | **ml-distance**（纯 JS cosine） | MIT | hnswlib/chroma 需原生构建/重型服务，违背一键复现 |
| 聚类 | spectral-forge vs ml-kmeans | **spectral-forge** 主 + 引擎内 k-means 基线 | MIT | 你已有、8/8 不变量、非凸优于 k-means |
| 分类/重排 | forest-forge vs xgboost.js | **forest-forge** | MIT | 你已有、OOB/重要性齐备；xgboost.js 生态弱 |
| 评测 | ml-metrics（公共 npm 已不可得） vs 自研 | **自研标准实现**（标注 sklearn 语义） | MIT | 环境 registry 无法解析 ml-metrics；标准公式自研以保证可复现 |
| 编排 | 自研 DAG vs langchainjs | **自研薄编排** | — | 固定 pipeline，langchainjs 引入重依赖且抽象过度 |

## 5. 复现性与可验证性

- **引擎隔离**：forge 引擎经 `vm` 受控上下文加载，DOM-free，与浏览器解耦，可在 Node 无头运行并交叉验证。
- **可复现随机**：所有随机过程（RF bootstrap、k-means++、聚类）由 seed 驱动；相同 seed 结果 bit 级一致。
- **依赖锁定**：`package-lock.json` 锁定全部运行依赖；可选 SOTA 后端不写入 lock，保证 `npm ci` 零原生依赖可复现。
- **每模块独立验证**：`src/<mod>/<mod>.test.mjs` 单测 + `src/<mod>/index.mjs` 即最小可运行示例。

## 6. 已知限制（详见验收报告）

- tfidf 后端在短中文语料上主题区分弱（属"朴素基线"水平）；稠密 MiniLM 后端为可选升级路径（需联网安装原生依赖）。
- 向量索引为精确线性扫描，适用于 demo / 中小规模；超大规模应替换为 ANN（hnswlib）索引。
