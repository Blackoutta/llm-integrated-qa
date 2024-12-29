from loguru import logger
import json
import os
from typing import *
import re
from ._model import InferenceModel
from enum import Enum
from tqdm import tqdm
from ._prompt import *
import copy
import pandas as pd
from dataloader import DataLoader
import traceback
from ._answer_generator_type1 import AnswerGeneratorType1
from ._answer_generator_type2 import AnswerGeneratorType2
from ._answer_generator_type3 import AnswerGeneratorType3
from ._answer_generator_sql import AnswerGeneratorSql
from ._answer_generator import AnswerGenerator
from pathlib import Path
from vllm import LLM


class ClsResult(Enum):
    # 公司基本信息, type 1
    COMPANY_BASIC_INFO = "A"
    # 员工信息, type 1
    EMPLOYEE_INFO = "B"
    # 财务报表相关信息, type 1
    FINANCIAL_REPORT = "C"
    # 计算题, type 2
    CALCULATION = "D"
    # 统计题, type 2
    STATISTICS = "E"
    # 开放性问题
    OPEN = "F"  # type 3
    # 未知问题
    UNKNOWN = "G"  # type 1

    """
    type 1: 基本信息处理
    type 2: 统计计算
    type 3: 总结推理
    """

    @staticmethod
    def from_value(value: str):
        for member in ClsResult:
            v = value.strip().upper().encode("utf-8").decode("utf-8")
            if member.value == v:
                return member
        logger.error(f"Unknown cls result value: '{v}', encode: {v}")
        return ClsResult.UNKNOWN

    @staticmethod
    def is_member(value: str):
        return value in [member.value for member in ClsResult]


class Inferencer:
    def __init__(
        self,
        ctx_dir: Path,
        inference_dir: Path,
        cls_model: InferenceModel = None,
        keywords_model: InferenceModel = None,
        nl2sql_model: InferenceModel = None,
        generic_model: InferenceModel = None,
        default_model: InferenceModel = None,
    ):
        self.cls_model = cls_model
        self.keywords_model = keywords_model
        self.nl2sql_model = nl2sql_model
        self.default_model = default_model
        self.generic_model = generic_model

        self.dataloader = DataLoader(ctx_dir=ctx_dir, inference_dir=inference_dir)

        self.ctx_dir: Path = ctx_dir
        self.inference_dir: Path = inference_dir

        self.classification_map = {}
        self.keywords_map = {}
        self.nl2sql_map = {}

        self.inference_dir.mkdir(parents=True, exist_ok=True)

    def _get_related_companies(self, question):
        question = re.sub(r"[\(\)（）]", "", question)

        related_companies = []
        for k, md in self.dataloader.load_pdf_metadata_map().items():
            company = md["company"]
            abbr = md["abbr"]
            if company in question:
                related_companies.append(company)
            if abbr in question:
                related_companies.append(abbr)
        return related_companies

    def do_classification(
        self, questions: List[Dict], persist=False, unload_on_done=True
    ) -> List[Dict]:
        model = self.cls_model or self.default_model

        entries = []
        for q in tqdm(
            questions, total=len(questions), desc="inferencing for classification"
        ):
            question_id = q["id"]
            question = q["question"]

            related_comp_names = self._get_related_companies(question)

            model_answer = model.chat(classify_prompt(question), max_tokens=1)
            cls_result = ClsResult.from_value(model_answer)

            if re.findall("(状况|简要介绍|简要分析|概述|具体描述|审计意见)", question):
                cls_result = ClsResult.OPEN

            if re.findall("(什么是|指什么|什么意思|定义|含义|为什么)", question):
                cls_result = ClsResult.OPEN

            if model_answer in ["A", "B", "C", "D"] and len(related_comp_names) == 0:
                cls_result = ClsResult.OPEN

            if model_answer in ["E"] and len(related_comp_names) > 0:
                cls_result = ClsResult.UNKNOWN

            entry = {"id": question_id, "question": question, "class": cls_result.value}
            entries.append(entry)
            if persist:
                out_file_path = os.path.join(self.inference_dir, "classification.jsonl")
                Inferencer.dump_as_jsonl([entry], out_file_path)
        if unload_on_done:
            model.unload()
        self.classification_map = {e["id"]: e["class"] for e in entries}
        return entries

    def do_keywords_generation(
        self,
        questions: List[Dict],
        persist=False,
        lora_name="",
        unload_on_done=False,
    ) -> List[Dict]:
        model = self.keywords_model or self.default_model
        entries = []
        for q in tqdm(questions, total=len(questions), desc="inferencing for keywords"):
            question_id = q["id"]
            question = q["question"]

            model_answer = model.chat(
                keywords_prompt(question), max_tokens=128, lora_name=lora_name
            )
            keywords = model_answer.strip("```").split(",")
            keywords = [kw.strip() for kw in keywords]
            if len(keywords) == 0:
                logger.warning("问题{}的关键词为空".format(question["question"]))

            entry = {"id": question_id, "question": question, "keywords": keywords}
            entries.append(entry)

            if persist:
                out_file_path = os.path.join(self.inference_dir, "keywords.jsonl")
                Inferencer.dump_as_jsonl([entry], out_file_path)
        if unload_on_done:
            model.unload()
        self.keywords_map = {e["id"]: e["keywords"] for e in entries}
        return entries

    def do_sql_generation(
        self,
        questions: List[Dict],
        persist=False,
        lora_name="",
        unload_on_done=False,
    ) -> List[Dict]:
        model = self.nl2sql_model or self.default_model
        entries = []

        classification_map = (
            self.classification_map
            if len(self.classification_map) > 0
            else self.dataloader.load_classification_map()
        )
        logger.debug(classification_map)

        for q in tqdm(questions, total=len(questions), desc="inferencing for nl2sql"):
            question_id = q["id"]
            question = q["question"]

            clssification = classification_map.get(question_id)
            if clssification is None:
                logger.warning("问题{}没有分类结果".format(question_id))
                entries.append({"id": question_id, "question": question, "sql": None})
                continue
            if clssification != ClsResult.STATISTICS.value:
                entries.append({"id": question_id, "question": question, "sql": None})
                continue

            model_answer = model.chat(
                nl2sql_prompt(question), max_tokens=2200, lora_name=lora_name
            )
            entry = {"id": question_id, "question": question, "sql": model_answer}
            entries.append(entry)

            if persist:
                out_file_path = os.path.join(self.inference_dir, "nl2sql.jsonl")
                Inferencer.dump_as_jsonl([entry], out_file_path)
        if unload_on_done:
            model.unload()
        self.nl2sql_map = {e["id"]: e["sql"] for e in entries}
        return entries

    def do_answer_generation(
        self,
        questions: List[Dict],
        persist=False,
        unload_on_done=True,
    ) -> List[Dict]:
        model = self.generic_model or self.default_model
        out_file_path = os.path.join(self.inference_dir, "answers.jsonl")

        ag1 = AnswerGeneratorType1(dataloader=self.dataloader, model=model)
        ag2 = AnswerGeneratorType2(dataloader=self.dataloader, model=model)
        ag3 = AnswerGeneratorType3(dataloader=self.dataloader, model=model)
        ag_sql = AnswerGeneratorSql(dataloader=self.dataloader, model=model)

        switcher: Dict[str, AnswerGenerator] = {
            "A": ag1,
            "B": ag1,
            "C": ag1,
            "D": ag2,
            "E": ag_sql,
            "F": ag3,
            "G": ag1,
        }

        entries = []
        for q in tqdm(questions, desc="Answer Generation", total=len(questions)):
            question_id = q["id"]
            question = q["question"]

            question_type = ag1.get_question_type(question_id)
            generator = switcher[question_type]
            if generator is None:
                logger.error("问题{}没有分类结果".format(question_id))
                continue

            try:
                answer = generator.generate_answer(question_id, question, question_type)

            except Exception as e:
                traceback.print_exc()

            logger.debug(f"问题{question_id}的答案为: '{answer}'")
            entry = {"id": question_id, "question": question, "answer": answer}
            entries.append(entry)

            if persist:
                Inferencer.dump_as_jsonl([entry], out_file_path)
        if unload_on_done:
            model.unload()
        return entries

    @staticmethod
    def dump_as_jsonl(entries, file_path):
        with open(file_path, "a", encoding="utf-8") as f:
            for entry in entries:
                js = json.dumps(entry, ensure_ascii=False)
                f.write(js + "\n")
