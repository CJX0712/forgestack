// engine.mjs — 加载自 engine.core.js（算法引擎已在 vm 隔离环境验证）
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

export const SPEC = ctx['SPEC'];
export default ctx['SPEC'];
