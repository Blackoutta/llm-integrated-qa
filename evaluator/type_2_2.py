from .datatypes import *
from text2vec import semantic_search


class Type_2_2_Evaluator(TypedEvaluator):
    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        gt_txt = gt.answer
        pred_txt = pred.answer

        prompt = gt.prompt
        key_word = prompt.key_word

        key_word_list = prompt.key_word.split("、")
        key_value = prompt.prom_answer

        tmp_count = 0
        for key_word in key_word_list:
            if prompt.extra[key_word] in pred_txt:
                tmp_count += 1

        score = 0.0
        if key_value == "相同" and key_value in pred_txt and "不相同" not in pred_txt:
            score += 0.25
            score += (
                semantic_search(
                    self.sentence_model.encode([pred_txt]),
                    self.sentence_model.encode(gt_txt),
                    top_k=1,
                )[0][0]["score"]
                * 0.5
            )
            if tmp_count == len(key_word_list):
                score += 0.25
        elif key_value == "不相同" and key_value in pred_txt:
            score += 0.25
            score += (
                semantic_search(
                    self.sentence_model.encode([pred_txt]),
                    self.sentence_model.encode(gt_txt),
                    top_k=1,
                )[0][0]["score"]
                * 0.5
            )
            if tmp_count == len(key_word_list):
                score += 0.25
        return score
