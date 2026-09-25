// src/orchestrator/corpus.mjs — 内置多主题评测语料（3 主题 × 8 篇）
// 用于演示检索 / 聚类 / 重排全链路，并作为量化基线的 ground truth（topic 字段）。
// Author: 晨星. License: MIT.
export const DOCS = [
  // ---- 机器学习 (ml) ----
  { id: 'ml01', topic: 'ml', text: '神经网络通过反向传播算法学习权重，梯度下降最小化损失函数。' },
  { id: 'ml02', topic: 'ml', text: 'Transformer 用自注意力机制捕捉序列中 token 之间的长距离依赖。' },
  { id: 'ml03', topic: 'ml', text: '随机森林由多棵决策树组成，通过 bagging 降低方差提升泛化。' },
  { id: 'ml04', topic: 'ml', text: '过拟合发生在模型记住训练集噪声，正则化与早停可缓解该问题。' },
  { id: 'ml05', topic: 'ml', text: 'Embedding 把离散 token 映射到稠密向量空间，相似语义距离更近。' },
  { id: 'ml06', topic: 'ml', text: '谱聚类先构造相似度图再对拉普拉斯矩阵做特征分解以发现非凸结构。' },
  { id: 'ml07', topic: 'ml', text: '交叉熵是分类任务常用的损失函数，与最大似然估计等价。' },
  { id: 'ml08', topic: 'ml', text: '梯度消失问题在深层网络中常见，残差连接缓解了这一现象。' },
  // ---- 生物 (bio) ----
  { id: 'bio01', topic: 'bio', text: 'DNA 双螺旋结构由碱基对通过氢键配对形成遗传信息载体。' },
  { id: 'bio02', topic: 'bio', text: '神经元通过突触传递电信号，构成神经系统信息处理的基本单元。' },
  { id: 'bio03', topic: 'bio', text: '光合作用在叶绿体中进行，将光能转化为化学能储存于糖分子。' },
  { id: 'bio04', topic: 'bio', text: '免疫系统通过淋巴细胞识别并清除入侵的病原体与异常细胞。' },
  { id: 'bio05', topic: 'bio', text: '蛋白质由氨基酸折叠而成，其三维结构决定生物功能。' },
  { id: 'bio06', topic: 'bio', text: '酶作为生物催化剂降低反应活化能，维持代谢途径高效运行。' },
  { id: 'bio07', topic: 'bio', text: '进化通过自然选择使适应环境的性状在种群中逐渐累积。' },
  { id: 'bio08', topic: 'bio', text: '基因表达受转录因子调控，决定细胞在特定条件下的功能。' },
  // ---- 历史 (history) ----
  { id: 'hi01', topic: 'history', text: '工业革命以蒸汽机为代表，使英国率先完成从手工到机械生产的转型。' },
  { id: 'hi02', topic: 'history', text: '文艺复兴以人文主义为核心，重塑了欧洲的艺术与科学精神。' },
  { id: 'hi03', topic: 'history', text: '丝绸之路连接欧亚，促进了贸易与文化的跨文明交流。' },
  { id: 'hi04', topic: 'history', text: '法国大革命推翻君主专制，提出自由平等博爱的政治理念。' },
  { id: 'hi05', topic: 'history', text: '冷战是美苏两大阵营在地缘政治与意识形态上的长期对峙。' },
  { id: 'hi06', topic: 'history', text: '罗马帝国以法律与道路网络维系了横跨三大洲的治理。' },
  { id: 'hi07', topic: 'history', text: '印刷术的普及大幅降低了知识传播成本，推动了宗教改革。' },
  { id: 'hi08', topic: 'history', text: '启蒙运动强调理性与科学，为现代民主制度奠定思想基础。' },
];

// 评测用查询（标注真实主题，用于计算 recall@k / NDCG@k）
export const QUERIES = [
  { q: '神经网络如何避免过拟合？', topic: 'ml' },
  { q: 'Transformer 的自注意力机制原理是什么？', topic: 'ml' },
  { q: 'DNA 与基因表达如何决定细胞功能？', topic: 'bio' },
  { q: '神经元和突触怎样传递信号？', topic: 'bio' },
  { q: '工业革命为何率先发生在英国？', topic: 'history' },
  { q: '丝绸之路对欧亚文明交流有何影响？', topic: 'history' },
];
