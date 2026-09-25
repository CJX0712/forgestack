// src/common/rng.mjs — 可复现随机数与数据划分（M0）
// Author: 晨星. License: MIT.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffle(arr, rng = Math.random) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
export function splitIndices(n, frac, rng = Math.random) {
  const idx = shuffle([...Array(n).keys()], rng);
  const k = Math.max(1, Math.floor(n * frac));
  return { train: idx.slice(0, k), test: idx.slice(k) };
}
