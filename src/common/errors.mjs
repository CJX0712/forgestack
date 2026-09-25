// src/common/errors.mjs — 统一错误码与错误类型（M0）
// Author: 晨星. License: MIT.
export const ErrCode = {
  TOK_EMPTY: 'TOK_EMPTY',
  TOK_BAD_MERGE: 'TOK_BAD_MERGE',
  TOK_UNTRAINED: 'TOK_UNTRAINED',
  EMB_EMPTY: 'EMB_EMPTY',
  EMB_LOAD: 'EMB_LOAD',
  VS_DIM_MISMATCH: 'VS_DIM_MISMATCH',
  VS_EMPTY: 'VS_EMPTY',
  CLU_K_INVALID: 'CLU_K_INVALID',
  CLU_SINGULAR: 'CLU_SINGULAR',
  RNK_UNFIT: 'RNK_UNFIT',
  RNK_INSUFFICIENT: 'RNK_INSUFFICIENT',
  EVAL_NO_GT: 'EVAL_NO_GROUND_TRUTH',
  ORC_NOT_INGESTED: 'ORC_NOT_INGESTED',
  ORC_EMPTY_QUERY: 'ORC_EMPTY_QUERY',
  CFG_INVALID: 'CFG_INVALID',
};

export class ForgeError extends Error {
  constructor(code, message, details = {}) {
    super(message || code);
    this.name = 'ForgeError';
    this.code = code;
    this.details = details;
  }
  toJSON() {
    return { error: this.code, message: this.message, details: this.details };
  }
}
