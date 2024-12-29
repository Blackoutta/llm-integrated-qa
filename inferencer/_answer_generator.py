from loguru import logger
import re
from typing import *
from dataloader import DataLoader
from difflib import SequenceMatcher
from ._prompt import type1_prompt, keyword_extraction_prompt_type3, type3_prompt
from ._model import InferenceModel
import itertools
from fastbm25 import fastbm25
from ._answer_generator_util import AnswerGeneratorUtil
from abc import ABC, abstractmethod


valid_table_map = {
    "A": ["basic_info", "no_table"],
    "B": ["employee_info", "dev_info", "no_table"],
    "C": ["cbs_info", "cscf_info", "cis_info", "no_table"],
    "G": [
        "basic_info",
        "employee_info",
        "dev_info",
        "cbs_info",
        "cscf_info",
        "cis_info",
        "no_table",
    ],
}


class AnswerGenerator(ABC):
    def __init__(self, dataloader: DataLoader, model: InferenceModel):
        self.dataloader = dataloader
        self.model = model
        self.valid_table_map = valid_table_map

    def get_match_pdf_names(self, question):
        years = AnswerGeneratorUtil.extract_years(question)
        match_keys = []
        for k, v in self.dataloader.load_pdf_metadata_map().items():
            company = v["company"]
            abbr = v["abbr"]
            year = v["year"].replace("年", "").replace(" ", "")
            if company in question and year in years:
                match_keys.append(k)
            if abbr in question and year in years:
                match_keys.append(k)
        match_keys = list(set(match_keys))
        # 前面已经完全匹配了年份, 所以可以删除年份
        overlap_len = [
            len(
                AnswerGeneratorUtil.get_matching_substrs(x, re.sub("\d?", "", question))
            )
            for x in match_keys
        ]
        match_keys = sorted(
            zip(match_keys, overlap_len), key=lambda x: x[1], reverse=True
        )
        if len(match_keys) > 1:
            # 多个结果重合率完全相同
            if len(set([t[1] for t in match_keys])) == 1:
                pass
            else:
                logger.warning("匹配到多个结果{}".format(match_keys))
                match_keys = match_keys[:1]
        match_keys = [k[0] for k in match_keys]
        return match_keys

    def get_company_name_and_abbr_code_of_question(self, pdf_keys, ori_question):
        company_infos = []
        md_map = self.dataloader.load_pdf_metadata_map()
        for pdf_key in pdf_keys:
            company_infos.append(
                (
                    md_map[pdf_key]["company"],
                    md_map[pdf_key]["abbr"],
                    md_map[pdf_key]["code"],
                )
            )
        company = ""
        abbr = ""
        code = ""
        real_comp = ""
        if len(company_infos) == 0:
            return company, abbr, code, real_comp, False

        company = company_infos[0][0]
        abbr = company_infos[0][1]
        code = company_infos[0][2]
        real_comp = company if company in ori_question else abbr
        return company, abbr, code, real_comp, True

    def get_question_type(self, question_id) -> str:
        return self.dataloader.load_classification_map().get(question_id, "F")

    def get_question_keywords(self, question_id):
        return self.dataloader.load_keywords_map().get(question_id, [])

    def recall_annual_report_texts(
        self,
        anoy_question: str,
        keywords: str,
        key,
    ):
        anoy_question = re.sub(r"(公司|年报|根据|数据|介绍)", "", anoy_question)
        logger.debug("anoy_question: {}".format(anoy_question.replace("<", "")))

        text_pages = self.dataloader.load_pdf_pages(key)
        text_lines = list(itertools.chain(*[page.split("\n") for page in text_pages]))
        text_lines = [line for line in text_lines if len(line) > 0]
        if len(text_lines) == 0:
            return []
        model = fastbm25(text_lines)
        result_keywords = model.top_k_sentence(keywords, k=3)
        result_question = model.top_k_sentence(anoy_question, k=3)
        top_match_indexes = [t[1] for t in result_question + result_keywords]
        block_line_indexes = AnswerGeneratorUtil.merge_idx(
            top_match_indexes, len(text_lines), 0, 30
        )

        text_blocks = [
            "\n".join([text_lines[idx] for idx in line_indexes])
            for line_indexes in block_line_indexes
        ]
        text_blocks = [re.sub(" {3,}", "\t", text_block) for text_block in text_blocks]

        text_blocks = [
            (
                t,
                SequenceMatcher(None, anoy_question, t, autojunk=False)
                .find_longest_match()
                .size,
            )
            for t in text_blocks
        ]

        max_match_size = max([t[1] for t in text_blocks])
        text_blocks = [t[0] for t in text_blocks if t[1] == max_match_size]

        if sum([len(t) for t in text_blocks]) > 2000:
            max_avg_len = int(2000 / len(text_blocks))
            text_blocks = [t[:max_avg_len] for t in text_blocks]

        text_blocks = [AnswerGeneratorUtil.rewrite_text_block(t) for t in text_blocks]
        text_blocks = ["```\n{}\n```".format(t) for t in text_blocks]
        return text_blocks

    def parse_question_keywords(self, question, real_company, years):
        question = (
            re.sub(r"[\(\)（）]", "", question)
            .replace("为？", "是什么？")
            .replace("是？", "是什么？")
            .replace("为多少", "是多少")
        )
        anoy_question = AnswerGeneratorUtil.anoy_question_xx(
            question, real_company, years
        )
        anoy_question = re.sub(
            r"(XX公司|XXXX年|XXXX|保留两位小数|对比|相比|报告期内|哪家|上市公司|第[1234567890一二三四五六七八九十]+[高低]|最[高低](的|的前|的后)?[1234567890一二三四五六七八九十]+家)",
            "",
            anoy_question,
        )
        if anoy_question[0] == "的":
            anoy_question = anoy_question[1:]
        answer = self.model(keyword_extraction_prompt_type3(anoy_question))

        key_words = AnswerGeneratorUtil.parse_keyword_from_answer(anoy_question, answer)
        # 无法提取，删除的再试一次
        if len(key_words) == 0:
            anoy_question = anoy_question.replace("的", "")
            answer = self.model(keyword_extraction_prompt_type3(anoy_question))
            key_words = AnswerGeneratorUtil.parse_keyword_from_answer(
                anoy_question, answer
            )
        if len(key_words) == 0:
            logger.debug("无法提取关键词")
            key_words = [anoy_question]

        return anoy_question, key_words

    @staticmethod
    def cleanup_question(question) -> str:
        return re.sub(r"[\(\)（）]", "", question)

    @abstractmethod
    def generate_answer(self, question_id, question, question_type) -> str:
        pass
