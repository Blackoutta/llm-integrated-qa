from .datatypes import *
from text2vec import semantic_search
import numpy as np


class Type_1_2_Evaluator(TypedEvaluator):
    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        gt_txt = gt.answer
        pred_txt = pred.answer
        prompt = gt.prompt
        key_word = prompt.key_word
        key_word_list = prompt.key_word.split("、")

        correct_value_cnt = 0

        for key_word in key_word_list:
            value2check = prompt.extra[key_word]
            if value2check in pred_txt:
                correct_value_cnt += 1

        accuracy_score = 1 * (correct_value_cnt / len(key_word_list))
        semantic_score = semantic_search(
            self.sentence_model.encode([pred_txt]),
            self.sentence_model.encode(gt_txt),
            top_k=1,
        )[0][0]["score"]

        return np.average([accuracy_score, semantic_score], weights=[0.75, 0.25])
