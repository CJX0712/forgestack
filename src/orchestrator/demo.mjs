// src/orchestrator/demo.mjs — 端到端演示 + 量化基线报告
// 运行：npm run demo（默认 tfidf 后端，零依赖确定性） | npm run demo:sota（MiniLM SOTA 后端）
// Author: 晨星. License: MIT.
import { Pipeline } from './index.mjs';
import { DOCS, QUERIES } from './corpus.mjs';
import { logger } from '../common/logger.mjs';

function parseArgs() {
  const args = process.argv.slice(2);
  const backend = args.includes('--backend') ? args[args.indexOf('--backend') + 1] : 'tfidf';
  return { backend };
}

function bar(label, rows, headers) {
  const w = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i]).length)));
  const line = (cells) => '| ' + cells.map((c, i) => String(c).padEnd(w[i])).join(' | ') + ' |';
  const sep = '+' + w.map((x) => '-'.repeat(x + 2)).join('+') + '+';
  console.log(sep);
  console.log(line(headers));
  console.log(sep);
  for (const r of rows) console.log(line(r));
  console.log(sep);
  console.log('');
}

async function main() {
  const { backend } = parseArgs();
  console.log('══════════════════════════════════════════════════════════════');
  console.log('  ForgeStack · 模块化本地 AI 系统 — 端到端 Demo');
  console.log('  Author: 晨星 (CJX0712) | backend =', backend);
  console.log('══════════════════════════════════════════════════════════════');

  const pipe = new Pipeline({ backend, seed: 42, k: 3, retrieveK: 8 });
  await pipe.ingest(DOCS);

  // ---- 1. 系统信息 ----
  const st = pipe.stats;
  bar('系统信息', [
    ['docs', st.nDocs],
    ['embedding backend', st.backend],
    ['embedding dims', st.dims],
    ['clusters (k)', st.clusters],
    ['ranker features', 'sim / lexical / sameCluster / lenRatio'],
    ['ranker importances', st.rankerImportances ? st.rankerImportances.join(', ') : 'n/a'],
  ], ['item', 'value']);

  // ---- 1.5 嵌入吞吐基线（texts/sec） ----
  const allTexts = DOCS.map((d) => d.text).concat(QUERIES.map((q) => q.q));
  const t0 = performance.now();
  await pipe.embedder.embed(allTexts);
  const dt = (performance.now() - t0) / 1000;
  const tput = allTexts.length / dt;
  bar('嵌入吞吐基线', [
    ['backend', pipe.embedBackend],
    ['文本数', allTexts.length],
    ['耗时(ms)', (dt * 1000).toFixed(1)],
    ['吞吐(texts/sec)', tput.toFixed(1)],
  ], ['item', 'value']);

  // ---- 2. 聚类基准：谱聚类 vs k-means（合成非凸） ----
  const bench = pipe.benchmarkClustering();
  bar('聚类基准 (accuracy, 越高越好)', [
    ['moons (两月牙, 非凸)', bench.moons.spectral.toFixed(3), bench.moons.kmeans.toFixed(3)],
    ['circles (同心环, 非凸)', bench.circles.spectral.toFixed(3), bench.circles.kmeans.toFixed(3)],
  ], ['dataset', 'spectral', 'kmeans']);

  // ---- 3. 文档主题发现：谱聚类 vs k-means ----
  const top = pipe.topics();
  bar('文档主题发现 (topic 恢复 accuracy)', [
    ['spectral (NJW)', top.spectralAcc.toFixed(3)],
    ['k-means (基线)', top.kmeansAcc.toFixed(3)],
    ['laplacian 对称误差', top.laplacianSymErr.toExponential(2)],
  ], ['method', 'value']);

  // ---- 4. 检索 + 重排量化 ----
  const idxOf = new Map(DOCS.map((d, i) => [d.id, i]));
  const kVals = [3, 5];
  const agg = { vec: { r: [0, 0], n: [0, 0], h: [0, 0] }, rf: { r: [0, 0], n: [0, 0], h: [0, 0] } };
  const qrows = [];
  for (const q of QUERIES) {
    const rel = DOCS.filter((d) => d.topic === q.topic).map((d) => d.id);
    const relSet = new Set(rel);
    const grade = {};
    rel.forEach((id) => (grade[id] = 1));
    const res = await pipe.query(q.q, 5);
    const vecOrder = res.retrieved.map((r) => r.id);
    const rfOrder = res.reranked.map((r) => r.id);
    const row = [q.q.slice(0, 14)];
    for (let ki = 0; ki < kVals.length; ki++) {
      const k = kVals[ki];
      const rv = pipe.evaluator.recallAtK(rel, vecOrder, k);
      const nv = pipe.evaluator.ndcgAtK(vecOrder, grade, k);
      const rr = pipe.evaluator.recallAtK(rel, rfOrder, k);
      const nr = pipe.evaluator.ndcgAtK(rfOrder, grade, k);
      const hv = vecOrder.slice(0, k).some((id) => relSet.has(id)) ? 1 : 0;
      const hr = rfOrder.slice(0, k).some((id) => relSet.has(id)) ? 1 : 0;
      agg.vec.r[ki] += rv; agg.vec.n[ki] += nv; agg.vec.h[ki] += hv;
      agg.rf.r[ki] += rr; agg.rf.n[ki] += nr; agg.rf.h[ki] += hr;
      row.push(`R${k}:${rr.toFixed(2)}/N${k}:${nr.toFixed(2)}`);
    }
    qrows.push(row);
  }
  bar('查询检索+重排 (per-query, R=recall N=NDCG, RF 重排后)', qrows, [
    'query', 'k=3', 'k=5',
  ]);
  const nQ = QUERIES.length;
  bar('检索聚合 (均值) — 向量 vs RF重排', [
    ['recall@3', (agg.vec.r[0] / nQ).toFixed(3), (agg.rf.r[0] / nQ).toFixed(3)],
    ['recall@5', (agg.vec.r[1] / nQ).toFixed(3), (agg.rf.r[1] / nQ).toFixed(3)],
    ['ndcg@3', (agg.vec.n[0] / nQ).toFixed(3), (agg.rf.n[0] / nQ).toFixed(3)],
    ['ndcg@5', (agg.vec.n[1] / nQ).toFixed(3), (agg.rf.n[1] / nQ).toFixed(3)],
    ['hit@3 (命中率)', (agg.vec.h[0] / nQ).toFixed(3), (agg.rf.h[0] / nQ).toFixed(3)],
    ['hit@5 (命中率)', (agg.vec.h[1] / nQ).toFixed(3), (agg.rf.h[1] / nQ).toFixed(3)],
  ], ['metric', '向量检索', 'RF重排']);

  // ---- 5. 分词器（token 预算） ----
  const ts = pipe.tokenizer.stats();
  bar('Tokenizer (bpe-forge)', [
    ['baseChars', ts.chars],
    ['tokens (final)', ts.tokens],
    ['compression ratio', ts.ratio.toFixed(3)],
    ['bytes / token', ts.bytesPerToken.toFixed(3)],
    ['vocabSize', pipe.tokenizer.vocabSize],
  ], ['item', 'value']);

  // ---- 6. 示例回答 ----
  console.log('示例回答（首个查询，RF 重排后 top1）：');
  const ex = await pipe.query(QUERIES[0].q, 5);
  console.log('  Q:', QUERIES[0].q);
  console.log('  A:', ex.answer);
  console.log('');
  console.log('✅ Demo 跑通：ingest → embed → retrieve → cluster → rerank → answer 全链路 0 手工干预。');
}

main().catch((e) => {
  logger.error('demo 失败', { code: e.code, message: e.message });
  console.error(e);
  process.exit(1);
});
