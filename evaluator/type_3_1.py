from .datatypes import *
from text2vec import semantic_search
import re
import numpy as np


class Type_3_1_Evaluator(TypedEvaluator):
    score_on_empty_keywords: bool = True
    score_on_true_negative: bool = True

    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        gt_txt = gt.answer
        pred_txt = pred.answer
        key_word = gt.prompt.key_word

        key_word_list = key_word.split("、")
        key_length = len(key_word_list)
        keyword_hit = 0

        for kw in key_word_list:
            if re.search(kw, pred_txt):
                keyword_hit += 1
        accuracy_score = keyword_hit / key_length

        semantic_score = semantic_search(
            self.sentence_model.encode([pred_txt]),
            self.sentence_model.encode(gt_txt),
            top_k=1,
        )[0][0]["score"]
        return np.average([accuracy_score, semantic_score], weights=[0.1, 0.9])
