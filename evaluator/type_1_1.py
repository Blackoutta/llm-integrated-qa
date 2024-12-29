from .datatypes import *
from text2vec import semantic_search
import numpy as np


class Type_1_1_Evaluator(TypedEvaluator):
    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        prompt = gt.prompt
        key_word = prompt.key_word

        value2check = prompt.extra[key_word]
        accuracy_score = 1 if value2check in pred.answer else 0.0

        semantic_score = semantic_search(
            self.sentence_model.encode([pred.answer]),
            self.sentence_model.encode(gt.answer),
            top_k=1,
        )[0][0]["score"]
        return np.average([accuracy_score, semantic_score], weights=[0.75, 0.25])
