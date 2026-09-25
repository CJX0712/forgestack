// scripts/extract-engines.mjs
// 从桌面三个 forge 仓库的 index.html 中抽取 <script id="engine"> 算法引擎，
// 落为自包含、可复现的 engine.core.js（纯算法源码）+ engine.mjs（ESM 加载器，vm 隔离运行）。
// 引擎层为 DOM-free，README 已声明；用 vm 提供受控上下文，避免任何浏览器全局依赖。
// Author: 晨星. License: MIT.
import { readFileSync, writeFileSync } from 'node:fs';
import { createContext, runInContext } from 'node:vm';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const SOURCES = [
  {
    html: 'C:/Users/Administrator/Desktop/bpe-forge/index.html',
    mod: 'tokenizer',
    global: 'BPE',
    probe: (g) => g.train(g.CORPUS ? g.CORPUS : 'ab', 5),
  },
  {
    html: 'C:/Users/Administrator/Desktop/spectral-forge/index.html',
    mod: 'clustering',
    global: 'SPEC',
    probe: (g) => g.makeMoons(20, 20, 2),
  },
  {
    html: 'C:/Users/Administrator/Desktop/forest-forge-v2/index.html',
    mod: 'ranker',
    global: 'FOREST',
    probe: (g) => g.makeDataset('xor', 20, 1, 0),
  },
];

// 受控上下文：仅暴露算法引擎所需的语言内置与标准全局，模拟"无浏览器环境"。
function buildContext() {
  const ctx = {
    console, Math, JSON, Object, Array, Map, Set, String, Number, Boolean, RegExp,
    Date, TextEncoder, TextDecoder, Uint8Array, Int8Array, Int16Array, Int32Array,
    Uint16Array, Float32Array, Float64Array, Uint8ClampedArray, BigInt,
    isFinite, isNaN, parseInt, parseFloat, Infinity, NaN,
    structuredClone, Proxy, Reflect, Symbol, Promise, Error, TypeError, RangeError,
    performance,
  };
  ctx.globalThis = ctx;
  return ctx;
}

function extract(htmlPath, globalName) {
  const html = readFileSync(htmlPath, 'utf8');
  const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
  if (!m) throw new Error(`engine script not found in ${htmlPath}`);
  return m[1];
}

function runEngine(src, globalName) {
  const ctx = buildContext();
  createContext(ctx);
  runInContext(src, ctx, { filename: 'engine.core.js' });
  return ctx[globalName];
}

let ok = true;
for (const s of SOURCES) {
  try {
    const src = extract(s.html, s.global);
    const modDir = join(ROOT, 'src', s.mod);
    writeFileSync(join(modDir, 'engine.core.js'), src, 'utf8');
    // 验证：在 vm 中加载并执行探针，确认引擎可用
    const g = runEngine(src, s.global);
    if (!g) throw new Error(`global ${s.global} not set`);
    s.probe(g);
    // 写 ESM 加载器
    const loader = `// engine.mjs — 加载自 engine.core.js（算法引擎已在 vm 隔离环境验证）
// Author: 晨星. License: MIT.
import { readFileSync } from 'node:fs';
import { createContext, runInContext } from 'node:vm';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(__dirname, 'engine.core.js'), 'utf8');

function buildContext() {
  const ctx = {
    console, Math, JSON, Object, Array, Map, Set, String, Number, Boolean, RegExp,
    Date, TextEncoder, TextDecoder, Uint8Array, Int8Array, Int16Array, Int32Array,
    Uint16Array, Float32Array, Float64Array, Uint8ClampedArray, BigInt,
    isFinite, isNaN, parseInt, parseFloat, Infinity, NaN,
    structuredClone, Proxy, Reflect, Symbol, Promise, Error, TypeError, RangeError,
    performance,
  };
  ctx.globalThis = ctx;
  return ctx;
}

const ctx = buildContext();
createContext(ctx);
runInContext(SRC, ctx, { filename: 'engine.core.js' });

export const ${s.global} = ctx['${s.global}'];
export default ctx['${s.global}'];
`;
    writeFileSync(join(modDir, 'engine.mjs'), loader, 'utf8');
    console.log(`✅ ${s.mod}: extracted engine, global=${s.global}, bytes=${src.length}, probe OK`);
  } catch (e) {
    ok = false;
    console.error(`❌ ${s.mod}: ${e.message}`);
  }
}
process.exit(ok ? 0 : 1);
