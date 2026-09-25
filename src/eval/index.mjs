// src/eval/index.mjs — M7 Eval 接口层
// 复用 common/metrics.mjs 的标准度量实现（ml-metrics 在本环境 registry 不可得，已书面说明）。
// Author: 晨星. License: MIT.
import * as M from '../common/metrics.mjs';

export class Evaluator {
  recallAtK(relevantIds, retrievedIds, k) {
    return M.recallAtK(relevantIds, retrievedIds, k);
  }
  ndcgAtK(retrievedIds, gradeMap, k) {
    return M.ndcgAtK(retrievedIds, gradeMap, k);
  }
  precisionRecallF1(predictedSet, relevantSet) {
    return M.precisionRecallF1(predictedSet, relevantSet);
  }
  accuracy(predicted, truth) {
    return M.accuracy(predicted, truth);
  }
  nmi(trueLabels, predLabels) {
    return M.nmi(trueLabels, predLabels);
  }
}
export default Evaluator;
