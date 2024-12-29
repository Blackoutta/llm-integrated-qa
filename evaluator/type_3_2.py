from .datatypes import *
from text2vec import semantic_search


class Type_3_2_Evaluator(TypedEvaluator):
    score_on_empty_keywords: bool = False
    score_on_true_negative: bool = False

    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        gt_txt = gt.answer
        pred_txt = pred.answer
        return semantic_search(
            self.sentence_model.encode([pred_txt]),
            self.sentence_model.encode(gt_txt),
            top_k=1,
        )[0][0]["score"]
