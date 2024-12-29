from ._answer_generator import AnswerGenerator
from dataloader import DataLoader
from ._model import InferenceModel
from ._answer_generator_util import AnswerGeneratorUtil
import re
from loguru import logger
from ._prompt import type3_prompt


class AnswerGeneratorType3(AnswerGenerator):
    def __init__(self, dataloader: DataLoader, model: InferenceModel):
        super().__init__(dataloader, model)

    def generate_answer(self, question_id, question, question_type) -> str:
        question_keywords = self.get_question_keywords(question_id)
        ori_question = AnswerGenerator.cleanup_question(question)
        mactched_pdf_names = self.get_match_pdf_names(ori_question)
        years = AnswerGeneratorUtil.extract_years(ori_question)
        _, _, _, real_comp, is_year_valid = (
            self.get_company_name_and_abbr_code_of_question(
                mactched_pdf_names, ori_question
            )
        )
        answer = ""
        if len(years) == 0:
            answer = self.model(ori_question)
            return answer
        if not is_year_valid:
            logger.error(
                f"年报匹配失败, question_type: {question_type},  years: {years}, matched_pdf_names: {mactched_pdf_names}, question: {ori_question}"
            )
            return ""

        anoy_question, _ = self.parse_question_keywords(ori_question, real_comp, years)
        logger.debug(f"问题关键词: {question_keywords}")

        background = "***************{}{}年年报***************\n".format(
            real_comp, years[0]
        )
        matched_text = self.recall_annual_report_texts(
            anoy_question,
            "".join(question_keywords),
            mactched_pdf_names[0],
        )
        for block_idx, text_block in enumerate(matched_text):
            background += "{}片段:{}{}\n".format("-" * 15, block_idx + 1, "-" * 15)
            background += text_block
            background += "\n"
        prompt = type3_prompt(
            background,
            ori_question,
        )
        logger.debug(f"type 3 prompt for question: {question_id}: {prompt}")

        if len(prompt) > 5120:
            prompt = prompt[:5120]
        answer = self.model(prompt)
        return answer
