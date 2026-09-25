
(function(){
  "use strict";
  var SPEC = {};
  SPEC.version = "1.0.0";

  // ---------- RNG ----------
  function mulberry32(a){
    return function(){
      a|=0; a=a+0x6D2B79F5|0;
      var t=Math.imul(a^a>>>15,1|a);
      t=t+Math.imul(t^t>>>7,61|t)^t;
      return ((t^t>>>14)>>>0)/4294967296;
    };
  }
  SPEC.rng = mulberry32;
  function randn(rng){ var u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

  // ---------- 向量运算 ----------
  function dot(a,b){ var s=0; for(var i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
  function norm(a){ return Math.sqrt(dot(a,a)); }
  function matVec(M,v){ var n=M.length, r=new Float64Array(n);
    for(var i=0;i<n;i++){ var s=0,row=M[i]; for(var j=0;j<n;j++) s+=row[j]*v[j]; r[i]=s; } return r; }
  function randUnit(n,seed){ var rng=mulberry32(seed), v=new Float64Array(n), s=0;
    for(var i=0;i<n;i++){ v[i]=rng()*2-1; s+=v[i]*v[i]; } s=Math.sqrt(s);
    for(var k=0;k<n;k++) v[k]/=s; return v; }

  // ---------- 数据集 ----------
  SPEC.makeBlobs=function(n1,n2,sep,seed){
    var rng=mulberry32(seed||1), X=[],y=[];
    var c=[[-sep,0],[sep,0]];
    for(var i=0;i<n1;i++){ X.push([c[0][0]+randn(rng)*0.3, c[0][1]+randn(rng)*0.3]); y.push(0); }
    for(var j=0;j<n2;j++){ X.push([c[1][0]+randn(rng)*0.3, c[1][1]+randn(rng)*0.3]); y.push(1); }
    return {X:X,y:y};
  };
  SPEC.makeMoons=function(n1,n2,seed){
    var rng=mulberry32(seed||2), X=[],y=[];
    for(var i=0;i<n1;i++){ var t=Math.PI*(i/n1); X.push([Math.cos(t)+randn(rng)*0.1, Math.sin(t)+randn(rng)*0.1]); y.push(0); }
    for(var j=0;j<n2;j++){ var u=Math.PI*(j/n2); X.push([1-Math.cos(u)+randn(rng)*0.1, -Math.sin(u)-0.5+randn(rng)*0.1]); y.push(1); }
    return {X:X,y:y};
  };
  SPEC.makeCircles=function(n1,n2,seed){
    var rng=mulberry32(seed||3), X=[],y=[];
    for(var i=0;i<n1;i++){ var t=2*Math.PI*(i/n1); X.push([Math.cos(t)*(1+randn(rng)*0.08), Math.sin(t)*(1+randn(rng)*0.08)]); y.push(0); }
    for(var j=0;j<n2;j++){ var u=2*Math.PI*(j/n2); X.push([Math.cos(u)*(3+randn(rng)*0.12), Math.sin(u)*(3+randn(rng)*0.12)]); y.push(1); }
    return {X:X,y:y};
  };

  // ---------- 距离与相似图 ----------
  function dist2(a,b){ var s=0; for(var i=0;i<a.length;i++){ var d=a[i]-b[i]; s+=d*d; } return s; }
  function median(arr){ var a=arr.slice().sort(function(x,y){return x-y;}); var m=a.length>>1;
    return a.length%2 ? a[m] : (a[m-1]+a[m])/2; }
  SPEC.buildSimilarity=function(X,opt){
    opt=opt||{}; var n=X.length, mode=opt.mode||'knn', k=opt.k||10, sigma=opt.sigma;
    var sig=sigma;
    if(sig===undefined||sig===null||isNaN(sig)){
      var ds=[];
      for(var i=0;i<n;i++){ var best=Infinity; for(var j=0;j<n;j++) if(j!==i){ var d=dist2(X[i],X[j]); if(d<best)best=d; } ds.push(Math.sqrt(best)); }
      sig=2*median(ds)+1e-9;
    }
    var W=[]; for(var a=0;a<n;a++) W.push(new Float64Array(n));
    var i,j;
    if(mode==='full'){
      for(i=0;i<n;i++) for(j=i+1;j<n;j++){ var w=Math.exp(-dist2(X[i],X[j])/(sig*sig)); W[i][j]=w; W[j][i]=w; }
    } else if(mode==='eps'){
      var eps=opt.eps||sig*2;
      for(i=0;i<n;i++) for(j=i+1;j<n;j++){ var d2=dist2(X[i],X[j]); if(d2<eps*eps){ var w2=Math.exp(-d2/(sig*sig)); W[i][j]=w2; W[j][i]=w2; } }
    } else {
      for(i=0;i<n;i++){
        var idx=[]; for(j=0;j<n;j++) if(j!==i) idx.push([dist2(X[i],X[j]),j]);
        idx.sort(function(p,q){return p[0]-q[0];});
        for(var t=0;t<k && t<idx.length;t++){ var jj=idx[t][1]; var w3=Math.exp(-idx[t][0]/(sig*sig)); W[i][jj]=w3; W[jj][i]=w3; }
      }
    }
    return {W:W, sig:sig, mode:mode, k:k};
  };
  SPEC.degree=function(W){ var n=W.length, d=new Float64Array(n);
    for(var i=0;i<n;i++){ var s=0; for(var j=0;j<n;j++) s+=W[i][j]; d[i]=s; } return d; };
  SPEC.normLaplacian=function(W){
    var n=W.length, d=SPEC.degree(W);
    var M=[]; for(var i=0;i<n;i++) M.push(new Float64Array(n));
    for(i=0;i<n;i++){ var di=1/(Math.sqrt(d[i])+1e-12);
      for(var j=0;j<n;j++){ var dj=1/(Math.sqrt(d[j])+1e-12); M[i][j]=W[i][j]*di*dj; } }
    return {M:M,d:d};
  };

  // ---------- Lanczos：求对称矩阵 M 的前 m 个最大特征对 ----------
  SPEC.lanczosEigs=function(M,n,m,seed){
    seed=seed||12345;
    var steps=m+6, v=randUnit(n,seed), V=[v], alpha=[], beta=[0];
    var w=matVec(M,v), j=0;
    for(; j<steps; j++){
      var a=dot(w, V[j]); alpha.push(a);
      var w2=new Float64Array(n);
      for(var i=0;i<n;i++) w2[i]=w[i]-a*V[j][i];
      if(j>0){ var bj=beta[j]; for(var i2=0;i2<n;i2++) w2[i2]-=bj*V[j-1][i2]; }
      // full reorthogonalization (2-pass modified Gram-Schmidt) — robust for clustered spectra
      for(var pass=0; pass<2; pass++){
        for(var c=0; c<V.length; c++){
          var proj=dot(w2, V[c]);
          for(var i3=0;i3<n;i3++) w2[i3]-=proj*V[c][i3];
        }
      }
      var b=norm(w2); beta.push(b);
      if(b<1e-12) break;
      var vNext=new Float64Array(n);
      for(var i4=0;i4<n;i4++) vNext[i4]=w2[i4]/b;
      V.push(vNext); w=matVec(M,vNext);
    }
    var mUsed=alpha.length;
    var T=[]; for(var r=0;r<mUsed;r++) T.push(new Float64Array(mUsed));
    for(var p=0;p<mUsed;p++){ T[p][p]=alpha[p]; if(p+1<mUsed){ T[p][p+1]=beta[p+1]; T[p+1][p]=beta[p+1]; } }
    var je=jacobiEig(T,mUsed);
    var order=je.vals.map(function(val,idx){return [val,idx];}).sort(function(x,y){return y[0]-x[0];});
    var topM=Math.min(m,mUsed), outVals=[], outVecs=[];
    for(var t=0;t<topM;t++){
      var idx=order[t][1]; outVals.push(order[t][0]);
      var coef=je.vecs[idx], rv=new Float64Array(n);
      for(var c2=0;c2<mUsed;c2++){ var vc=coef[c2], col=V[c2]; for(var ii=0;ii<n;ii++) rv[ii]+=vc*col[ii]; }
      outVecs.push(rv);
    }
    return {vals:outVals, vecs:outVecs, n:n, mUsed:mUsed};
  };

  // ---------- Jacobi 全特征值求解（交叉验证用） ----------
  function jacobiEig(A,n){
    var a=[], v=[], i, j;
    for(i=0;i<n;i++){ a.push(Float64Array.from(A[i]));
      var row=new Float64Array(n); row[i]=1; v.push(row); }
    var maxSweeps=100*n;
    for(var sweep=0; sweep<maxSweeps; sweep++){
      var p=0,q=1,maxv=0;
      for(i=0;i<n;i++) for(j=i+1;j<n;j++){ var val=Math.abs(a[i][j]); if(val>maxv){ maxv=val; p=i; q=j; } }
      if(maxv<1e-14) break;
      var app=a[p][p], aqq=a[q][q], apq=a[p][q];
      var phi=0.5*Math.atan2(2*apq, aqq-app);
      var c=Math.cos(phi), s=Math.sin(phi);
      for(i=0;i<n;i++){ var aip=a[i][p], aiq=a[i][q]; a[i][p]=c*aip-s*aiq; a[i][q]=s*aip+c*aiq; }
      for(i=0;i<n;i++){ var api=a[p][i], aqi=a[q][i]; a[p][i]=c*api-s*aqi; a[q][i]=s*api+c*aqi; }
      for(i=0;i<n;i++){ var vip=v[i][p], viq=v[i][q]; v[i][p]=c*vip-s*viq; v[i][q]=s*vip+c*viq; }
    }
    var vals=[]; for(i=0;i<n;i++) vals.push(a[i][i]);
    var vecs=[]; for(j=0;j<n;j++){ var col=new Float64Array(n); for(i=0;i<n;i++) col[i]=v[i][j]; vecs.push(col); }
    return {vals:vals, vecs:vecs};
  }
  SPEC.jacobiEig=jacobiEig;

  // ---------- k-means（确定性，种子化） ----------
  function kmeans(X,k,seed){
    var n=X.length, dim=X[0].length, rng=mulberry32(seed||7), cent=[], i, j, d;
    cent.push(X[Math.floor(rng()*n)].slice());
    var d2=new Float64Array(n);
    for(var c=1;c<k;c++){
      var tot=0;
      for(i=0;i<n;i++){ var best=Infinity; for(j=0;j<cent.length;j++){ var dd=dist2(X[i],cent[j]); if(dd<best) best=dd; } if(best<d2[i]) d2[i]=best; tot+=d2[i]; }
      var r=rng()*tot, acc=0, pick=0;
      for(i=0;i<n;i++){ acc+=d2[i]; if(acc>=r){ pick=i; break; } }
      cent.push(X[pick].slice());
    }
    var assign=new Int32Array(n);
    for(var iter=0; iter<80; iter++){
      var changed=false;
      for(i=0;i<n;i++){ var bj=0,best2=Infinity; for(j=0;j<k;j++){ var dd2=dist2(X[i],cent[j]); if(dd2<best2){ best2=dd2; bj=j; } } if(assign[i]!==bj){ assign[i]=bj; changed=true; } }
      var sum=[]; for(var cc=0;cc<k;cc++){ var arr=new Float64Array(dim); sum.push(arr); }
      var cnt=new Int32Array(k);
      for(i=0;i<n;i++){ var a2=assign[i]; cnt[a2]++; for(d=0;d<dim;d++) sum[a2][d]+=X[i][d]; }
      for(j=0;j<k;j++){ if(cnt[j]>0) for(d=0;d<dim;d++) cent[j][d]=sum[j][d]/cnt[j]; }
      if(!changed && iter>0) break;
    }
    return {assign:Array.prototype.slice.call(assign), centroids:cent};
  }
  SPEC.kmeans=kmeans;

  // ---------- 主流程：Ng–Jordan–Weiss 谱聚类 ----------
  SPEC.spectralCluster=function(X,opt){
    opt=opt||{}; var k=opt.k||2;
    var sim=opt.sim || SPEC.buildSimilarity(X,{mode:opt.mode||'knn', k:opt.knn||10, sigma:opt.sigma});
    var W=sim.W, nl=SPEC.normLaplacian(W), M=nl.M;
    // Exact full eigendecomposition. Graph sizes here are <=~120 nodes, so O(n^3) is instant and
    // numerically exact — the right tool for clustered/near-degenerate spectra where plain Lanczos
    // (even with reorthogonalization) loses accuracy. Lanczos remains available as a scalable
    // alternative and is cross-validated in _smoke on a well-conditioned matrix.
    // NJW: the k eigenvectors for the k smallest L_sym eigenvalues == the top-k of M.
    var jac=SPEC.jacobiEig(M, X.length);
    var order=jac.vals.map(function(val,idx){return [val,idx];}).sort(function(a,b){return b[0]-a[0];});
    var U=[];
    for(var t=0;t<k;t++) U.push(jac.vecs[order[t][1]]);
    var lz=SPEC.lanczosEigs(M, X.length, k, opt.seed||999);
    var rows=[];
    for(var i=0;i<X.length;i++){
      var s=0, row=new Float64Array(k);
      for(var c=0;c<k;c++){ row[c]=U[c][i]; s+=row[c]*row[c]; }
      s=Math.sqrt(s)+1e-12;
      for(var c2=0;c2<k;c2++) row[c2]/=s;
      rows.push(row);
    }
    var km=kmeans(rows, k, opt.seed||777);
    var embed=X.map(function(_,i){ return [U[0][i], (U[1]?U[1][i]:0)]; });
    return { clusters:km.assign, centroids:km.centroids, embedding:embed, M:M, W:W, d:nl.d,
             sim:sim, U:U, eigenvectors:U,
             jacobiVals:order.slice(0,k).map(function(o){return o[0];}),
             lanczos:lz };
  };

  // ---------- 准确率（置换匹配） ----------
  SPEC.clusterAccuracy=function(pred,trueLabels){
    var k=Math.max.apply(null,pred)+1, cnt=[], i, c, t;
    for(c=0;c<k;c++){ cnt.push(new Array(k).fill(0)); }
    for(i=0;i<pred.length;i++){ cnt[pred[i]][trueLabels[i]]++; }
    var usedT=new Array(k).fill(false), total=0;
    for(c=0;c<k;c++){ var best=-1,bj=-1;
      for(t=0;t<k;t++){ if(!usedT[t] && cnt[c][t]>best){ best=cnt[c][t]; bj=t; } }
      usedT[bj]=true; total+=best;
    }
    return {acc:total/pred.length, perm:usedT};
  };

  // ---------- 诊断 ----------
  SPEC.orthonormError=function(vecs){
    var m=vecs.length, maxErr=0;
    for(var i=0;i<m;i++) for(var j=0;j<m;j++){
      var v=dot(vecs[i],vecs[j]); if(i===j) v-=1;
      if(Math.abs(v)>maxErr) maxErr=Math.abs(v);
    }
    return maxErr;
  };
  SPEC.rayleigh=function(M,v){ var Mv=matVec(M,v); return dot(v,Mv)/dot(v,v); };
  SPEC.symmetryError=function(M){
    var e=0; for(var i=0;i<M.length;i++) for(var j=0;j<M.length;j++){ var d=M[i][j]-M[j][i]; if(Math.abs(d)>e) e=Math.abs(d); }
    return e;
  };

  if(typeof globalThis!=="undefined") globalThis.SPEC=SPEC;
})();
