from ._answer_generator import AnswerGenerator
from dataloader import DataLoader
from ._model import InferenceModel
from ._answer_generator_util import AnswerGeneratorUtil
import re
from loguru import logger
from ._prompt import type1_prompt


class AnswerGeneratorType1(AnswerGenerator):
    def __init__(self, dataloader: DataLoader, model: InferenceModel):
        super().__init__(dataloader, model)

    def generate_answer(self, question_id, question, question_type) -> str:
        question_keywords = self.get_question_keywords(question_id)

        ori_question = AnswerGenerator.cleanup_question(question)

        years = AnswerGeneratorUtil.extract_years(ori_question)
        mactched_pdf_names = self.get_match_pdf_names(ori_question)
        company, abbr, code, _, _ = self.get_company_name_and_abbr_code_of_question(
            mactched_pdf_names, ori_question
        )

        answer = "经查询，无法回答: {}".format(ori_question)

        logger.debug("问题关键词: {}".format(question_keywords))
        background = ""
        tot_matched_rows = []
        for year in years:
            # [(table name, row_year, column name, row_value)]
            data_rows = self.dataloader.find_company_table_data(company, [year])

            background += "已知{}(简称:{},证券代码:{}){}年的资料如下:\n    ".format(
                company, abbr, code, year
            )
            matched_table_rows = []
            for keyword in question_keywords:
                matched_table_rows.extend(
                    # [(table name, row_year, column name, row_value)]
                    AnswerGeneratorUtil.recall_pdf_tables(
                        keyword,
                        [year],
                        data_rows,
                        min_match_number=3,
                        valid_tables=self.valid_table_map[question_type],
                    )
                )

            if len(matched_table_rows) == 0:
                for table_row in data_rows:
                    if table_row[0] in self.valid_table_map[question_type]:
                        matched_table_rows.append(table_row)

            table_text = AnswerGeneratorUtil.table_to_text(
                matched_table_rows,
                with_year=False,
            )
            background += table_text
            background += "\n"

            tot_matched_rows.extend(matched_table_rows)

        tot_matched_rows = AnswerGeneratorUtil.add_text_compare_in_table(
            tot_matched_rows
        )
        tot_text = AnswerGeneratorUtil.table_to_text(tot_matched_rows, with_year=True)

        if "相同" in tot_text or "不相同且不同" in tot_text:
            answer = tot_text
        else:
            prompt = type1_prompt(ori_question, company, abbr, years).format(
                background, ori_question
            )
            if len(prompt) > 5120:
                prompt = prompt[:5120]
            logger.debug(f"type 1 prompt for question: {question_id}: {prompt}")
            answer = self.model(prompt)
        return answer
