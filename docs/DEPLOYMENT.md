# 部署指南（Deployment）

> 作者：晨星（CJX0712） · License：MIT

## 1. 环境要求

| 组件 | 版本 / 说明 |
|------|-------------|
| Node.js | ≥ 20（推荐 22 LTS，本交付基于 v22.22.2 验证） |
| npm | ≥ 10（用于 `npm ci` 锁版本安装） |
| 操作系统 | Windows / macOS / Linux（纯 JS，无原生编译） |
| 网络 | 默认路径**零网络依赖**（tfidf 后端离线可跑）；启用 MiniLM SOTA 后端需联网下载模型 |

## 2. 一键复现（干净环境）

```bash
git clone https://github.com/CJX0712/forgestack.git
cd forgestack
npm ci            # 基于 package-lock.json 干净重建依赖（仅 ml-distance，无原生编译）
npm test          # 41 项单测全绿
npm run demo      # 端到端 demo：摄取→嵌入→检索→聚类→重排→回答，零手工干预
```

验收要点（DoD）：`clone → npm ci → npm run demo` 三步即可跑通，**无需任何手动配置或环境变量**。

## 3. 可选 SOTA 稠密向量后端

默认 `tfidf` 后端确定性、离线可跑。要升级为业界 SOTA 稠密嵌入（all-MiniLM-L6-v2）：

```bash
npm i @xenova/transformers     # 或 transformers.js（可选安装，不写入 lock）
npm run demo:sota              # 以 --backend minilm 运行
```

- 首次运行会自动从 HuggingFace 下载模型（约 80MB），后续缓存复用。
- 该依赖**不写入 `package-lock.json`**，因此不影响默认路径的可复现性（缺失时 demo 会明确报错并给出安装提示）。
- 注意：本交付所在的受限沙箱因无法获取 `sharp`/`onnxruntime` 原生依赖而未能安装该后端；代码路径已就绪，可在具备完整网络与构建工具的环境启用。

## 4. 复现性与稳定性保证

- **依赖锁定**：`package-lock.json` 精确锁定全部运行依赖版本；`npm ci` 禁止版本漂移。
- **零原生编译**：运行依赖仅 `ml-distance`（纯 JS），避免 `node-gyp` / 预编译二进制导致的环境差异。
- **确定性**：引擎使用 seed 驱动 RNG；相同输入+seed 结果 bit 级一致。
- **统一错误码**：任一模块异常均抛 `ForgeError` 并带 `code`，便于运维排查。

## 5. 性能基线（本机实测）

| 场景 | 指标 | 值 |
|------|------|-----|
| 嵌入 | 吞吐 | 47,132 texts/sec（tfidf, 30 文本） |
| 聚类 | moons（非凸）accuracy | 谱聚类 1.000 vs k-means 0.963 |
| 聚类 | circles（非凸）accuracy | 谱聚类 1.000 vs k-means 0.500 |
| 检索 | NDCG@5 | 向量 0.580 → RF 重排 **0.655** |
| 检索 | Recall@5 | 向量 0.292 → RF 重排 **0.354** |
| 分词 | 压缩率 / 词表 | 0.846 / 404 |

## 6. 生产化建议（后续优化方向）

- 向量索引替换为 ANN（hnswlib / 自托管 Qdrant）以支持百万级文档。
- 检索阶段引入重排 Cross-Encoder（如 `sentence-transformers/ms-marco-MiniLM`）进一步提升 NDCG。
- 增加持久化（向量库落盘、docs 增量摄取）。
- 启用 MiniLM 稠密后端以获得语义级检索（短文本主题区分显著优于 tfidf）。
