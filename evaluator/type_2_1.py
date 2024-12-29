from .datatypes import *
from text2vec import semantic_search
import numpy as np


class Type_2_1_Evaluator(TypedEvaluator):
    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        gt_txt = gt.answer
        pred_txt = pred.answer
        prompt = gt.prompt
        gt_fact = prompt.prom_answer

        accuracy_score = 1.0 if gt_fact in pred_txt else 0
        semantic_score = semantic_search(
            self.sentence_model.encode([pred_txt]),
            self.sentence_model.encode(gt_txt),
            top_k=1,
        )[0][0]["score"]

        return np.average([accuracy_score, semantic_score], weights=[0.75, 0.25])
