
(function(){
"use strict";
// 字符级 BPE。约定：pair 的 map key 用 "\u0000" 连接两个 token；文本本身不应包含 \u0000（roundtrip 仍正确，仅合并记账可能有歧义）。
var CORPUS =
"Transformers 把注意力变成了通用接口。Attention is all you need，一句论文标题定义了整个时代。\n" +
"GPT、BERT、LLaMA、DeepSeek……这些名字背后是同一套骨架：tokenizer 把文本切成 token，embedding 把 token 变成向量，" +
"self-attention 让每个位置互相张望，feed-forward 负责整理行囊。\n" +
"分词器是模型的第一个决策：同样一句话，BPE 可以切成「大模型」「的」「前世今生」，也可以切成更碎的子词。" +
"切得越合并，序列越短，推理越便宜；切得越细，词表越大，embedding 越孤单。\n" +
"中文没有空格，英文有词界，代码里有驼峰与下划线——一个优秀的 tokenizer 必须同时伺候三位主人。\n" +
"Byte-level BPE 干脆把所有文本先当字节看：任何 unicode 都不会 OOV，GPT-2 与 Llama 3 都走这条路。" +
"代价是：一个汉字常常要占两个 token，中文用户的 API 账单因此悄悄变厚。\n" +
"本实验室从零训练一个字符级 BPE：看着 merge 表一行行长出来，压缩曲线一段段往下掉，" +
"你会亲眼看见「语言的结构」如何被频率统计一点点榨出来。\n" +
"注意力机制的本质是信息检索：query 去问，key 来对号，value 交答案，softmax 分配权重。\n" +
"位置编码告诉模型词序，残差连接保护梯度，layer normalization 稳住训练。\n" +
"预训练靠下一个词预测，微调靠指令对齐，RLHF 靠人类偏好，蒸馏靠大教小。\n" +
"参数从百万到万亿，上下文从五百到百万，价格却一年降十倍——规模定律仍在生效。\n" +
"Attention is all you need. Attention is efficient. Attention is sparse. Tokenizer 训练是压缩，压缩即智能。\n" +
"模型 architecture 决定上限，数据决定下限，tokenizer 决定成本。\n" +
"The model reads the token, the token feeds the model. The tokenizer is the interface between text and model.\n" +
"好的 tokenizer 让常用词更短，让稀有词不炸，让代码与中英文和平共处。";

function pairKey(a,b){ return a+"\u0000"+b; }
function splitKey(k){ var i=k.indexOf("\u0000"); return [k.slice(0,i), k.slice(i+1)]; }

// 训练：返回 {merges, baseChars, vocabSize, curve}
// merges[i] = {a,b,count}；curve[i] = 第 i 次 merge 后的 token 数（curve[0] 为初始字符数）
function train(text, numMerges){
  var seq = Array.from(text);
  var vocab = Object.create(null);
  for (var i=0;i<seq.length;i++) vocab[seq[i]] = 1;
  var baseChars = 0; for (var k in vocab) baseChars++;
  var merges = [];
  var curve = [seq.length];
  for (var m=0; m<numMerges; m++){
    var counts = Object.create(null);
    var order = [];
    for (var j=0; j+1<seq.length; j++){
      var pk = pairKey(seq[j], seq[j+1]);
      if (!(pk in counts)){ counts[pk] = 1; order.push(pk); }
      else counts[pk]++;
    }
    // 选最高频 pair；平票取字典序最小（保证确定性）
    var best = null, bestC = 1;
    for (var t=0; t<order.length; t++){
      var p2 = order[t], c2 = counts[p2];
      if (c2 < 2) continue;
      if (best === null || c2 > bestC || (c2 === bestC && p2 < best)){ best = p2; bestC = c2; }
    }
    if (best === null) break;
    var ab = splitKey(best);
    merges.push({ a: ab[0], b: ab[1], count: bestC });
    var out = [];
    for (var s=0; s<seq.length; s++){
      if (s+1<seq.length && seq[s]===ab[0] && seq[s+1]===ab[1]){ out.push(ab[0]+ab[1]); s++; }
      else out.push(seq[s]);
    }
    seq = out;
    curve.push(seq.length);
  }
  return { merges: merges, baseChars: baseChars, vocabSize: baseChars + merges.length, curve: curve };
}

// 编码：严格按 merge 训练顺序应用（标准 BPE 推理）
function encode(text, model){
  var seq = Array.from(text);
  for (var r=0; r<model.merges.length; r++){
    var a = model.merges[r].a, b = model.merges[r].b;
    var changed = false, out = [];
    for (var i=0; i<seq.length; i++){
      if (seq[i]===a && seq[i+1]===b){ out.push(a+b); i++; changed = true; }
      else out.push(seq[i]);
    }
    if (changed) seq = out;
  }
  return seq;
}

function decode(tokens){ return tokens.join(""); }

function stats(text, model){
  var toks = encode(text, model);
  var chars = Array.from(text).length;
  var bytes = null;
  if (typeof TextEncoder !== "undefined") bytes = new TextEncoder().encode(text).length;
  var uniq = Object.create(null), n = 0;
  for (var i=0;i<toks.length;i++){ if(!(toks[i] in uniq)){ uniq[toks[i]] = 1; n++; } }
  return {
    tokens: toks.length, chars: chars, bytes: bytes,
    ratio: chars ? toks.length/chars : 0,
    bytesPerToken: toks.length && bytes ? bytes/toks.length : 0,
    uniqTokens: n, vocabSize: model.vocabSize, merges: model.merges.length,
    toksArr: toks
  };
}

function tokenFreq(text, model){
  var toks = encode(text, model), f = Object.create(null);
  for (var i=0;i<toks.length;i++) f[toks[i]] = (f[toks[i]]||0)+1;
  var arr = [];
  for (var t in f) arr.push([t, f[t]]);
  arr.sort(function(x,y){ return y[1]-x[1] || (x[0]<y[0]?-1:1); });
  return arr;
}

globalThis.BPE = { CORPUS: CORPUS, train: train, encode: encode, decode: decode, stats: stats, tokenFreq: tokenFreq };
})();
