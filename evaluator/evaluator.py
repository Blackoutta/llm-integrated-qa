from typing import *
import json
import os
from pydantic import BaseModel
from text2vec import SentenceModel, semantic_search, Similarity
from pathlib import Path

from .datatypes import *
from .type_1_1 import Type_1_1_Evaluator
from .type_1_2 import Type_1_2_Evaluator
from .type_2_1 import Type_2_1_Evaluator
from .type_2_2 import Type_2_2_Evaluator
from .type_3_1 import Type_3_1_Evaluator
from .type_3_2 import Type_3_2_Evaluator
from loguru import logger
import numpy as np

file_dir = os.path.dirname(__file__)


class Evaluator(BaseModel):
    metrics_input: Path
    answer_input: Path = None
    eval_output: Path = None

    sentence_model_path: Path = Path(
        Path.home(), ".cache/modelscope/hub/Jerry0/text2vec-base-chinese"
    )

    sentence_model: SentenceModel = SentenceModel(
        model_name_or_path=sentence_model_path, device="cuda:0"
    )

    questions: List[Question] = []
    answers: Dict[int, Answer] = {}
    ground_truths: Dict[int, GroundTruth] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_questions(self) -> List[Question]:
        """
        加载问题集
        问题集文件示例：
        {"id": 0, "question": "无形资产是指什么？"}
        {"id": 1, "question": "请告诉我龙岩卓越新能源股份有限公司2021年的应付账款的具体数值"}
        {"id": 2, "question": "2019年负债总额第2高的上市公司是？"}
        {"id": 3, "question": "哪家上市公司，2019年净利润第十二高？"}
        """
        if len(self.questions) > 0:
            return self.questions
        results = []
        with open(os.path.join(self.metrics_input, "question.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                results.append(Question(**data))
            self.questions = results
        return results

    def load_answers(self) -> List[Answer]:
        if len(self.answers) > 0:
            return self.answers
        results = []
        with open(os.path.join(self.answer_input, "answers.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                results.append(Answer(**data))
        self.answers = results
        return results

    def load_ground_truths(self) -> List[GroundTruth]:
        if len(self.ground_truths) > 0:
            return self.ground_truths
        results = []
        with open(os.path.join(self.metrics_input, "ground_truth.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                results.append(GroundTruth(**data))
        self.ground_truths = results
        return results

    def answer_gt_pair(self) -> List[Tuple[Answer, GroundTruth]]:
        answers = self.load_answers()
        ground_truths = self.load_ground_truths()

        gt_dict = {gt.id: gt for gt in ground_truths}
        resutls = []
        for answer in answers:
            if answer.id not in gt_dict:
                raise ValueError(f"answer id {answer.id} not in ground truth")
            resutls.append((answer, gt_dict[answer.id]))
        return resutls

    def do_evaluation(self, persist=False) -> Dict:
        switcher: Dict[str, TypedEvaluator] = {
            "1": Type_1_1_Evaluator(sentence_model=self.sentence_model),
            "1-2": Type_1_2_Evaluator(sentence_model=self.sentence_model),
            "2-1": Type_2_1_Evaluator(sentence_model=self.sentence_model),
            "2-2": Type_2_2_Evaluator(sentence_model=self.sentence_model),
            "3-1": Type_3_1_Evaluator(sentence_model=self.sentence_model),
            "3-2": Type_3_2_Evaluator(sentence_model=self.sentence_model),
        }
        for answer, gt in self.answer_gt_pair():
            typed_evaluator = switcher[gt.type]
            if typed_evaluator is None:
                logger.warning(f"type {gt.type} not supported")
                continue
            logger.debug(f"evaluating question: {answer.id} with type {gt.type}")
            typed_evaluator.do_eval(answer, gt)

        s11 = switcher["1"].get_average_score()
        s12 = switcher["1-2"].get_average_score()
        s21 = switcher["2-1"].get_average_score()
        s22 = switcher["2-2"].get_average_score()
        s31 = switcher["3-1"].get_average_score()
        s32 = switcher["3-2"].get_average_score()

        final_score = np.average(
            [s11, s12, s21, s22, s31, s32],
            weights=[0.15, 0.15, 0.2, 0.2, 0.2, 0.1],
        )
        final_score = np.round(final_score, 4)

        score_dict = {
            "1-1": s11,
            "1-2": s12,
            "2-1": s21,
            "2-2": s22,
            "3-1": s31,
            "3-2": s32,
            "final_score": final_score,
        }
        if persist:
            self.eval_output.mkdir(parents=True, exist_ok=True)
            with open(Path(self.eval_output, "output.json"), "w") as f:
                json.dump(score_dict, f, ensure_ascii=False, indent=4)
        return score_dict


def test_evaluator():
    evaluator = Evaluator(
        metrics_input=Path(Path(file_dir).parent, "resources/metrics").as_posix(),
        answer_input=Path(Path(file_dir).parent, "resources/inferenced").as_posix(),
        eval_output=Path(Path(file_dir).parent, "resources/evaluated").as_posix(),
    )
    questions = evaluator.load_questions()
    assert len(questions) == 1000
    print(questions[0])

    answers = evaluator.load_answers()
    assert len(answers) > 0
    print(answers[0])

    ground_truths = evaluator.load_ground_truths()
    assert len(ground_truths) > 0
    print(ground_truths[22])

    evaluator.do_evaluation(persist=True)
