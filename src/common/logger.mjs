// src/common/logger.mjs — 极简分级日志（M0）
// Author: 晨星. License: MIT.
const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
let current = 20;
export function setLevel(name) {
  if (name in LEVELS) current = LEVELS[name];
}
function emit(levelName, msg, meta) {
  if (LEVELS[levelName] < current) return;
  const t = new Date().toISOString().slice(11, 23);
  const metaStr = meta ? ' ' + JSON.stringify(meta) : '';
  process.stdout.write(`[${levelName.toUpperCase()} ${t}] ${msg}${metaStr}\n`);
}
export const logger = {
  debug: (m, meta) => emit('debug', m, meta),
  info: (m, meta) => emit('info', m, meta),
  warn: (m, meta) => emit('warn', m, meta),
  error: (m, meta) => emit('error', m, meta),
  setLevel,
};
