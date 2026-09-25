// src/common/index.mjs — M0 公共件统一出口
// Author: 晨星. License: MIT.
export { ErrCode, ForgeError } from './errors.mjs';
export { logger, setLevel } from './logger.mjs';
export { mulberry32, shuffle, splitIndices } from './rng.mjs';
