
(function(){
  "use strict";

  // ---- 确定性随机数 ----
  function mulberry32(a){
    return function(){
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function randn(rng){
    var u=0,v=0;
    while(u===0) u=rng();
    while(v===0) v=rng();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  }
  function range(n){ var a=[]; for(var i=0;i<n;i++) a.push(i); return a; }

  // ---- 数据集生成（2 维特征，二分类）----
  function makeDataset(kind, n, seed, noise){
    if(noise==null) noise=0.06;
    var rng=mulberry32(seed>>>0);
    var X=[], y=[];
    var i, c, t, r, th, x, yy;
    for(i=0;i<n;i++){
      if(kind==="xor"){
        x  = (rng()*2-1)*1.9;
        yy = (rng()*2-1)*1.9;
        c  = ((x>=0)===(yy>=0))?0:1;              // 同号→0，异号→1（随机采样，有限样本天然不平衡→贪心树可学）
      } else if(kind==="blobs"){
        c = rng()<0.5?0:1;
        var mx = c? 1.25 : -1.25;
        x  = mx + 0.62*randn(rng);
        yy = 0  + 0.62*randn(rng);
      } else if(kind==="moons"){
        c = rng()<0.5?0:1;
        t = Math.PI*rng();
        if(c===0){ x = Math.cos(t); yy = Math.sin(t); }
        else      { x = 1 - Math.cos(t); yy = 0.55 - Math.sin(t); }
        x*=1.55; yy*=1.55;
      } else if(kind==="circles"){
        c = rng()<0.5?0:1;
        r = (c? 0.45 : 1.45) + 0.10*randn(rng);
        th = 2*Math.PI*rng();
        x = r*Math.cos(th); yy = r*Math.sin(th);
      } else if(kind==="spiral"){
        c = rng()<0.5?0:1;
        t = rng()*Math.PI*2.2;
        r = (i/n)*1.7 + 0.15;
        var off = c? Math.PI : 0;
        x  = r*Math.cos(t+off) + 0.12*randn(rng);
        yy = r*Math.sin(t+off) + 0.12*randn(rng);
        c = (c+ (i%2))%2; // 轻微打乱保证两类都有
      } else {
        c = rng()<0.5?0:1;
        x = randn(rng); yy = randn(rng);
      }
      x  += noise*randn(rng);
      yy += noise*randn(rng);
      X.push([x,yy]); y.push(c);
    }
    return {X:X, y:y, n:n, nFeat:2, classes:[0,1], kind:kind};
  }

  // ---- 计数 / Gini ----
  function classCounts(y, idx){
    var c={};
    for(var i=0;i<idx.length;i++){ var k=y[idx[i]]; c[k]=(c[k]||0)+1; }
    return c;
  }
  function giniFromCounts(counts, total){
    if(total===0) return 0;
    var g=0;
    for(var k in counts){ var p=counts[k]/total; g += p*(1-p); }
    return g;
  }
  function probFromCounts(counts, total){
    var p={};
    for(var k in counts) p[k]=counts[k]/total;
    return p;
  }
  function argmaxProb(p){
    var b=null, bv=-1;
    for(var k in p){ if(p[k]>bv){ bv=p[k]; b=+k; } }
    return b;
  }

  // ---- 特征子采样 ----
  function allFeats(d){ var a=[]; for(var i=0;i<d;i++) a.push(i); return a; }
  function sampleFeats(d, m, rng){
    var arr=[]; for(var i=0;i<d;i++) arr.push(i);
    for(var i=0;i<m;i++){
      var j=i+Math.floor(rng()*(d-i));
      var tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
    }
    return arr.slice(0,m).sort(function(a,b){return a-b;});
  }

  // ---- 单特征最佳分裂（增量扫描，O(n log n)）----
  function bestSplitForFeature(X, y, idx, f, parentGini){
    var order = idx.slice().sort(function(a,b){ return X[a][f]-X[b][f]; });
    var n=order.length;
    var countsR = classCounts(y, order);
    var countsL = {};
    var nR=n, nL=0;
    var best=null;
    for(var i=0;i<n-1;i++){
      var yi=y[order[i]];
      countsL[yi]=(countsL[yi]||0)+1; nL++;
      countsR[yi]=(countsR[yi]||0)-1; nR--; if(countsR[yi]<=0) delete countsR[yi];
      var vi=X[order[i]][f], vj=X[order[i+1]][f];
      if(vi===vj) continue;
      var gL=giniFromCounts(countsL,nL), gR=giniFromCounts(countsR,nR);
      var gain=parentGini - (nL/n)*gL - (nR/n)*gR;
      var thr=(vi+vj)/2;
      if(!best || gain>best.gain+1e-15) best={feat:f,thr:thr,gain:gain};
    }
    return best;
  }

  // ---- 暴力枚举最佳分裂（独立代码路径，交叉验证）----
  function bruteBestSplit(X, y, idx){
    var d = X[idx[0]].length;
    var best=0;
    for(var f=0; f<d; f++){
      var vals=[]; for(var i=0;i<idx.length;i++) vals.push(X[idx[i]][f]);
      vals.sort(function(a,b){return a-b;});
      var distinct=[];
      for(var i=0;i<vals.length;i++) if(i===0||vals[i]!==vals[i-1]) distinct.push(vals[i]);
      var parentGini = giniFromCounts(classCounts(y,idx), idx.length);
      for(var j=0;j<distinct.length-1;j++){
        var thr=(distinct[j]+distinct[j+1])/2;
        var cL={}, cR={}, nL=0, nR=0;
        for(var i=0;i<idx.length;i++){
          var ii=idx[i], yi=y[ii];
          if(X[ii][f]<=thr){ cL[yi]=(cL[yi]||0)+1; nL++; }
          else            { cR[yi]=(cR[yi]||0)+1; nR++; }
        }
        var gL=giniFromCounts(cL,nL), gR=giniFromCounts(cR,nR);
        var gain=parentGini - (nL/idx.length)*gL - (nR/idx.length)*gR;
        if(gain>best) best=gain;
      }
    }
    return best;
  }

  // ---- CART 递归建树 ----
  function buildTree(X, y, idx, depth, opts){
    var n = idx.length;
    var counts = classCounts(y, idx);
    var total = n;
    var gini = giniFromCounts(counts, total);
    var bestK=null, bestCount=0;
    for(var k in counts){ if(counts[k]>bestCount){ bestCount=counts[k]; bestK=+k; } }
    var node = {isLeaf:false, depth:depth, samples:n, gini:gini, majority:bestK, prob:probFromCounts(counts,total)};

    if(depth>=opts.maxDepth || n<=opts.minSamplesLeaf || gini<=1e-12 || n<opts.minSamplesSplit){
      node.isLeaf=true; return node;
    }
    var feats = (opts.mtry && opts.mtry < opts.nFeat) ? sampleFeats(opts.nFeat, opts.mtry, opts.rng) : allFeats(opts.nFeat);
    var best=null;
    for(var fi=0; fi<feats.length; fi++){
      var cand = bestSplitForFeature(X, y, idx, feats[fi], gini);
      if(cand && (!best || cand.gain>best.gain)) best=cand;
    }
    if(!best || best.gain<=1e-12){ node.isLeaf=true; return node; }

    var left=[], right=[];
    for(var i=0;i<idx.length;i++){
      var ii=idx[i];
      if(X[ii][best.feat] <= best.thr) left.push(ii); else right.push(ii);
    }
    node.feat=best.feat; node.thr=best.thr; node.gain=best.gain;
    node.left  = buildTree(X, y, left,  depth+1, opts);
    node.right = buildTree(X, y, right, depth+1, opts);
    return node;
  }

  // ---- 树预测 ----
  function treePredictProb(node, x){
    while(!node.isLeaf){ node = (x[node.feat] <= node.thr) ? node.left : node.right; }
    return node.prob;
  }
  function accuracyTree(tree, X, y){
    var c=0;
    for(var i=0;i<X.length;i++) if(argmaxProb(treePredictProb(tree, X[i]))===y[i]) c++;
    return c/X.length;
  }
  function allLeavesPure(node){
    if(node.isLeaf) return node.gini<=1e-9;
    return allLeavesPure(node.left) && allLeavesPure(node.right);
  }
  function cloneTree(node){ return JSON.parse(JSON.stringify(node)); }
  function walkStats(node){
    if(node.isLeaf) return {depth:node.depth, leaves:1};
    var l=walkStats(node.left), r=walkStats(node.right);
    return {depth:Math.max(l.depth,r.depth), leaves:l.leaves+r.leaves};
  }

  // ---- Random Forest ----
  function RandomForest(opts){
    opts = opts||{};
    var nTrees        = opts.nTrees||50;
    var maxDepth      = opts.maxDepth==null?10:opts.maxDepth;
    var minSamplesLeaf= opts.minSamplesLeaf==null?1:opts.minSamplesLeaf;
    var minSamplesSplit=opts.minSamplesSplit==null?2:opts.minSamplesSplit;
    var mtryMode      = opts.mtry||"sqrt";
    var seed          = opts.seed||1;
    var rng = mulberry32(seed>>>0);

    var trees=[], oobIdx=[], importances=[], nFeat=0, classes=[0,1], n=0, yTrain=null, Xref=null;

    function mtryFor(d){
      if(mtryMode==="sqrt") return Math.max(1, Math.floor(Math.sqrt(d)));
      return Math.min(d, Math.max(1, mtryMode|0));
    }
    function fit(X, y){
      n = X.length; nFeat = X[0].length; Xref=X; yTrain=y;
      var clsSet={}; for(var i=0;i<n;i++) clsSet[y[i]]=1;
      classes = Object.keys(clsSet).map(Number).sort(function(a,b){return a-b;});
      trees=[]; oobIdx=[]; importances=new Array(nFeat).fill(0);
      for(var t=0;t<nTrees;t++){
        var inbag=[], inSet={};
        for(var k=0;k<n;k++){ var s=Math.floor(rng()*n); inbag.push(s); inSet[s]=1; }
        var oob=[]; for(var i=0;i<n;i++) if(!inSet[i]) oob.push(i);
        var tree = buildTree(X, y, inbag, 0, {
          maxDepth:maxDepth, minSamplesLeaf:minSamplesLeaf, minSamplesSplit:minSamplesSplit,
          mtry:mtryFor(nFeat), nFeat:nFeat, rng:rng, classes:classes
        });
        trees.push(tree); oobIdx.push(oob);
        accumulateImportance(tree);
      }
      var total=0; for(var f=0;f<nFeat;f++) total+=importances[f];
      if(total>0){ for(var f=0;f<nFeat;f++) importances[f]/=total; }
      return this;
    }
    function accumulateImportance(node){
      if(node.isLeaf) return;
      importances[node.feat] += node.samples*node.gini
                              - node.left.samples*node.left.gini
                              - node.right.samples*node.right.gini;
      accumulateImportance(node.left); accumulateImportance(node.right);
    }
    function predictProbOne(x){
      var sum={};
      for(var t=0;t<trees.length;t++){
        var p=treePredictProb(trees[t], x);
        for(var c in p) sum[c]=(sum[c]||0)+p[c];
      }
      var k=0; for(var c in sum){ sum[c]/=trees.length; k++; }
      if(k===0){ for(var cc=0;cc<classes.length;cc++) sum[classes[cc]]=1/classes.length; }
      return sum;
    }
    function predictOne(x){ return argmaxProb(predictProbOne(x)); }
    function predict(X){ return X.map(predictOne); }
    function oobScore(){
      var err=0, cnt=0;
      for(var i=0;i<n;i++){
        var votes={}, nt=0;
        for(var t=0;t<trees.length;t++){
          if(oobIdx[t].indexOf(i)>=0){
            var p=treePredictProb(trees[t], Xref[i]);
            for(var c in p) votes[c]=(votes[c]||0)+p[c];
            nt++;
          }
        }
        if(nt===0) continue;
        if(argmaxProb(votes)!==yTrain[i]) err++;
        cnt++;
      }
      return cnt? 1-err/cnt : 0;
    }
    function treeStats(){
      var td=0, tl=0;
      for(var t=0;t<trees.length;t++){ var s=walkStats(trees[t]); td+=s.depth; tl+=s.leaves; }
      return {avgDepth: trees.length? td/trees.length:0, totalLeaves: tl, nTrees: trees.length};
    }
    return {
      fit:fit,
      predict:predict, predictOne:predictOne, predictProbOne:predictProbOne,
      oobScore:oobScore, treeStats:treeStats,
      get importances(){return importances;},
      get trees(){return trees;},
      get oobIdx(){return oobIdx;},
      get n(){return n;},
      get nFeat(){return nFeat;},
      get classes(){return classes;}
    };
  }

  // ---- 引擎自检：8 项不变量 ----
  function selfTest(){
    var res=[];
    var d = makeDataset("xor", 200, 42, 0);

    // 1. 确定性：同种子 → bit 级一致
    var f1 = new RandomForest({nTrees:30, maxDepth:8, minSamplesLeaf:1, mtry:2, seed:7}).fit(d.X,d.y);
    var f2 = new RandomForest({nTrees:30, maxDepth:8, minSamplesLeaf:1, mtry:2, seed:7}).fit(d.X,d.y);
    var p1=f1.predict(d.X), p2=f2.predict(d.X), det=true;
    for(var i=0;i<p1.length;i++) if(p1[i]!==p2[i]) det=false;
    res.push({name:"确定性（同种子预测 bit 级一致）", pass:det, detail: det?"identical":"mismatch"});

    // 2. XOR 无限深单树 100% 拟合
    var t = buildTree(d.X,d.y, range(d.n), 0,
      {maxDepth:8, minSamplesLeaf:1, minSamplesSplit:1, mtry:2, nFeat:2, rng:mulberry32(1), classes:[0,1]});
    var acc = accuracyTree(t, d.X, d.y);
    res.push({name:"XOR 单树 100% 拟合", pass: acc>=0.999, detail: acc.toFixed(4)});

    // 3. 叶子纯度 Gini=0
    var pure = allLeavesPure(t);
    res.push({name:"叶子纯度 Gini=0", pass:pure, detail: pure?"all pure":"impure"});

    // 4. 特征重要性求和=1
    var impSum=0; for(var f=0;f<f1.importances.length;f++) impSum+=f1.importances[f];
    res.push({name:"特征重要性求和=1", pass: Math.abs(impSum-1)<1e-9, detail: impSum.toFixed(12)});

    // 5. 袋外占比≈0.368
    var cov=0; for(var tt=0;tt<f1.oobIdx.length;tt++) cov+=f1.oobIdx[tt].length;
    cov = cov/(f1.oobIdx.length*f1.n);
    res.push({name:"袋外占比≈0.368", pass: Math.abs(cov-0.368)<0.05, detail: cov.toFixed(4)});

    // 6. 预测概率求和=1
    var pr=f1.predictProbOne(d.X[0]), ps=0; for(var c in pr) ps+=pr[c];
    res.push({name:"预测概率求和=1", pass: Math.abs(ps-1)<1e-9, detail: ps.toFixed(12)});

    // 7. 根分裂增益 == 暴力枚举
    var full = buildTree(d.X,d.y, range(d.n), 0,
      {maxDepth:8, minSamplesLeaf:1, minSamplesSplit:1, mtry:2, nFeat:2, rng:mulberry32(99), classes:[0,1]});
    var rootGain = full.gain;
    var brute = bruteBestSplit(d.X, d.y, range(d.n));
    res.push({name:"根分裂增益==暴力枚举", pass: Math.abs(rootGain-brute)<1e-9, detail:"Δ="+(rootGain-brute).toExponential(2)});

    // 8. 树克隆预测一致
    var tc = cloneTree(t);
    var accC = accuracyTree(tc, d.X, d.y);
    res.push({name:"树克隆预测一致", pass: accC===acc, detail: accC.toFixed(4)+" = "+acc.toFixed(4)});

    return res;
  }

  globalThis.FOREST = {
    version:"1.0.0",
    mulberry32:mulberry32,
    makeDataset:makeDataset,
    RandomForest:RandomForest,
    buildTree:buildTree,
    bestSplitForFeature:bestSplitForFeature,
    bruteBestSplit:bruteBestSplit,
    treePredictProb:treePredictProb,
    accuracyTree:accuracyTree,
    giniFromCounts:giniFromCounts,
    selfTest:selfTest
  };
})();
