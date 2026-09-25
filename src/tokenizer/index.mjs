// src/tokenizer/index.mjs — M1 Tokenizer 接口层（复用 bpe-forge 引擎）
// Author: 晨星. License: MIT.
import BPE from './engine.mjs';
import { ErrCode, ForgeError } from '../common/errors.mjs';

export class Tokenizer {
  constructor(opts = {}) {
    this.merges = opts.merges ?? 200;
    this.model = null;
    this.corpus = '';
  }
  train(corpus, merges = this.merges) {
    if (!corpus || typeof corpus !== 'string' || corpus.length === 0)
      throw new ForgeError(ErrCode.TOK_EMPTY, 'corpus 不能为空');
    if (!Number.isInteger(merges) || merges < 0)
      throw new ForgeError(ErrCode.TOK_BAD_MERGE, 'merge 次数必须为非负整数', { merges });
    this.corpus = corpus;
    this.model = BPE.train(corpus, merges);
    return this;
  }
  encode(text) {
    if (!this.model) throw new ForgeError(ErrCode.TOK_UNTRAINED, '分词器未训练，请先 train()');
    return BPE.encode(text, this.model);
  }
  decode(tokens) {
    return BPE.decode(tokens);
  }
  get vocabSize() {
    return this.model ? this.model.vocabSize : 0;
  }
  stats() {
    if (!this.model) throw new ForgeError(ErrCode.TOK_UNTRAINED, '分词器未训练');
    return BPE.stats(this.corpus, this.model);
  }
  tokenFreq() {
    if (!this.model) throw new ForgeError(ErrCode.TOK_UNTRAINED, '分词器未训练');
    return BPE.tokenFreq(this.corpus, this.model);
  }
}
export default Tokenizer;
export { BPE };
