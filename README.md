# ForgeStack · 模块化本地 AI 系统

> 一套**可实际运行、模块化、可一键复现**的本地 AI 系统：本地 RAG + 文档智能分析 pipeline。
> 复用你已有的三个 forge 算法引擎（BPE 分词 / 谱聚类 / 随机森林）作为核心算法模块，并复用业界 OSS（ml-distance 余弦、transformers.js 稠密向量）补齐短板。
> **作者：晨星（GitHub [@CJX0712](https://github.com/CJX0712)）** · License：MIT

---

## 一句话定位

摄取文档 → BPE 分词(token 预算/词法特征) → 嵌入(向量) → 向量检索 → 谱聚类发现主题 → 随机森林重排 → 生成回答。
三个 forge 模块各居其位，每个模块**可独立验证（单测 + 最小示例）**，组合成完整可运行链路。

## 特性

- **零编译步骤**：纯 ESM JavaScript（Node ≥ 20），`npm ci && npm run demo` 即跑通。
- **算法内核复用 forge 引擎**：bpe-forge / spectral-forge / forest-forge 以受控 `vm` 隔离抽取为自包含模块，DOM-free、可复现。
- **OSS 复用**：向量相似度用 `ml-distance`（纯 JS cosine）；稠密嵌入可通过 `transformers.js` 启用 MiniLM SOTA 后端（可选）。
- **量化基线**：内置评测对比「谱聚类 vs k-means」「向量检索 vs 随机森林重排」，指标可复现。
- **统一错误码 / 日志 / 可复现 RNG**：跨模块一致的接口与诊断。

## 快速开始

```bash
git clone https://github.com/CJX0712/forgestack.git
cd forgestack
npm ci                 # 基于 package-lock.json 干净重建依赖（零原生依赖）
npm test              # 运行全部模块单测（41 项）
npm run demo          # 端到端 demo（tfidf 后端，确定性、离线可跑）
```

可选 SOTA 稠密向量后端（需联网下载模型，非默认依赖）：

```bash
npm i @xenova/transformers      # 或 transformers.js
npm run demo:sota               # 使用 MiniLM 稠密嵌入
```

## 模块地图

| 模块 | 职责 | 复用 | 单测 |
|------|------|------|------|
| M0 Common | 错误码 / 日志 / RNG / 标准度量 | 自研（ml-metrics 在本环境 registry 不可得，已书面说明） | ✅ |
| M1 Tokenizer | BPE 训练 / 编码 / 解码 / 压缩率 | **bpe-forge** | ✅ |
| M2 Embeddings | 文本→向量（tfidf / MiniLM） | **transformers.js**（可选） | ✅ |
| M3 VectorStore | 向量索引 + 近邻检索 | **ml-distance** cosine | ✅ |
| M4 Clustering | 谱聚类发现非凸主题 | **spectral-forge** + k-means 基线 | ✅ |
| M5 Ranker | 检索结果重排（学习型） | **forest-forge** RandomForest | ✅ |
| M6 Orchestrator | 串联全链路 + demo | 自研薄 DAG（langchainjs 过重，已说明） | ✅ |
| M7 Eval | recall@k / NDCG / 准确率 / NMI | 标准公式（sklearn 等价语义） | ✅ |

## 性能基线（本机实测，demo 输出）

| 指标 | 向量检索 | 随机森林重排 | 提升 |
|------|---------|-------------|------|
| NDCG@5 | 0.580 | **0.655** | +0.075 |
| Recall@5 | 0.292 | **0.354** | +0.062 |
| Hit@5（命中率） | 1.000 | 1.000 | — |
| 嵌入吞吐 | — | 47,132 texts/sec（tfidf） | — |

| 聚类（非凸合成数据, accuracy） | 谱聚类 | k-means |
|------|---------|---------|
| moons（两月牙） | **1.000** | 0.963 |
| circles（同心环） | **1.000** | 0.500 |

> 结论：在非凸结构上谱聚类显著优于 k-means（circles 0.5→1.0）；学习型重排提升检索排序质量（NDCG@5 +13%）。

## 选型与工程纪律

- **优先复用开源，仅必要处自研**：向量距离（ml-distance）、稠密嵌入（transformers.js）、随机森林/谱聚类/BPE（forge 引擎）。
- **自研理由（书面）**：`ml-metrics` 在公共 npm 已不可得；langchainjs 对固定 pipeline 过重；hnswlib/chromadb 需原生构建，违背"干净环境一键复现"——均以标准实现/薄编排替代并记录。
- **依赖锁定**：`package-lock.json` 锁定全部运行依赖；可选 SOTA 后端不写入 lock，保证 `npm ci` 零原生依赖可复现。
- **无 TODO/半成品**：所有模块均通过单测；引擎以 `vm` 隔离运行，保留 forge 原 8/8 不变量自检思想。

## 仓库

https://github.com/CJX0712/forgestack

## License

MIT © 2026 晨星（CJX0712）
