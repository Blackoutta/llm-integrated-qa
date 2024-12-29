from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, List, Tuple
from abc import abstractmethod, ABC
from text2vec import SentenceModel, semantic_search
import numpy as np
from loguru import logger


class Question(BaseModel):
    id: int
    question: str


class Answer(BaseModel):
    id: int
    question: str
    answer: str


class Prompt(BaseModel):
    ent_short_name: Optional[str] = ""
    ent_name: Optional[str] = ""
    year: Optional[str] = ""
    key_word: Optional[str] = ""
    prom_answer: Optional[str] = ""
    extra: Dict[str, str] = {}  # 用于存储额外的字段

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.extra[key] = value

    # 动态设置属性
    def __setattr__(self, name, value):
        if name not in self.__dict__ and name not in self.extra:
            self.extra[name] = value
        else:
            super().__setattr__(name, value)

    # 动态获取属性
    def __getattr__(self, name):
        if name in self.extra:
            return self.extra[name]
        raise AttributeError(f"'Prompt' object has no attribute '{name}'")


class GroundTruth(BaseModel):
    id: int
    question: str
    prompt: Optional[Prompt] = {}
    answer: List[str]
    type: str


class TypedEvaluator(BaseModel, ABC):
    sentence_model: SentenceModel
    score_map: Dict[int, float] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    true_negative_keywords: str = "无|不|没有|未|否|非|莫|抱歉|毋"
    score_on_empty_keywords: bool = False
    score_on_true_negative: bool = True

    def do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        # cleanup
        pred.answer = TypedEvaluator.clean_up_txt(pred.answer)
        gt.answer = TypedEvaluator.clean_up_txt_list(gt.answer)
        for k, v in gt.prompt.extra.items():
            gt.prompt.extra[k] = TypedEvaluator.clean_up_txt(v)
        gt.prompt.prom_answer = TypedEvaluator.clean_up_txt(gt.prompt.prom_answer)

        # deal with empty keywords
        if self.score_on_empty_keywords and gt.prompt.key_word == "":
            logger.debug(f"question_id: {gt.id} is did empty keywords check")
            return semantic_search(
                self.sentence_model.encode([pred.answer]),
                self.sentence_model.encode(gt.answer),
                top_k=1,
            )[0][0]["score"]

        # deal with true negatives
        if self.score_on_true_negative:
            n_score, valid = self.score_true_negative(pred, gt)
            if valid:
                logger.debug(f"question_id: {gt.id} is did true negative check")
                return n_score

        logger.debug(f"question_id: {gt.id} did normal check")
        score = self._do_eval(pred, gt)
        self.score_map[pred.id] = score

    def get_average_score(self) -> float:
        scores = list(self.score_map.values())
        if len(scores) == 0:
            return 100.0
        return np.round(np.average(scores) * 100, 4)

    def score_true_negative(self, pred: Answer, gt: GroundTruth) -> Tuple[float, bool]:
        prompt = gt.prompt
        prompt_keyword = prompt.key_word
        if prompt_keyword != self.true_negative_keywords:
            return 0.0, False

        score = 0.0
        key_word_list = prompt_keyword.split("|")
        for kword in key_word_list:
            if kword in pred.answer:
                score += 0.25
                pred_txt = TypedEvaluator.clean_up_txt_list(pred.answer)
                score += (
                    semantic_search(
                        self.sentence_model.encode([pred_txt]),
                        self.sentence_model.encode(
                            TypedEvaluator.clean_up_txt_list(gt.answer)
                        ),
                        top_k=1,
                    )[0][0]["score"]
                    * 0.5
                )
                if prompt.year in pred_txt:
                    score += 0.25
                return score, True
        return 0.0, True

    @staticmethod
    def clean_up_txt_list(txt_list: List[str]) -> List[str]:
        return [TypedEvaluator.clean_up_txt(txt) for txt in txt_list]

    @staticmethod
    def clean_up_txt(txt: str) -> str:
        return txt.replace(",", "").replace(" ", "")

    @abstractmethod
    def _do_eval(self, pred: Answer, gt: GroundTruth) -> float:
        pass
