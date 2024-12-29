from ._answer_generator import AnswerGenerator
from dataloader import DataLoader
from ._model import InferenceModel
from ._answer_generator_util import AnswerGeneratorUtil
import re
from loguru import logger
from ._prompt import single_question_prompt


class AnswerGeneratorType2(AnswerGenerator):
    def __init__(self, dataloader: DataLoader, model: InferenceModel):
        super().__init__(dataloader, model)

    def generate_answer(self, question_id, question, question_type) -> str:
        question_keywords = self.get_question_keywords(question_id)

        ori_question = AnswerGenerator.cleanup_question(question)

        years = AnswerGeneratorUtil.extract_years(ori_question)
        mactched_pdf_names = self.get_match_pdf_names(ori_question)
        company, abbr, code, real_comp, is_valid = (
            self.get_company_name_and_abbr_code_of_question(
                mactched_pdf_names, ori_question
            )
        )

        if not is_valid:
            logger.error("匹配到了类别{}, 但是不存在报表".format(question_type))
            return ""

        if AnswerGeneratorUtil.is_type2_growth_rate(ori_question):
            offset_years = []
            for year in years:
                offset_years.extend([year, str(int(year) - 1)])

            pdf_table = self.dataloader.find_company_table_data(company, offset_years)
            pdf_table = AnswerGeneratorUtil.add_growth_rate_in_table(pdf_table)
        elif AnswerGeneratorUtil.is_type2_formula(ori_question):
            pdf_table = self.dataloader.find_company_table_data(company, years)
        else:
            logger.error("无法匹配, 该问题既不是增长率也不是公式计算")
            pdf_table = self.dataloader.find_company_table_data(company, years)

        (
            step_questions,
            step_keywords,
            variable_names,
            step_years,
            formula,
            question_formula,
        ) = AnswerGeneratorUtil.get_step_questions(
            ori_question,
            "".join(question_keywords),
            real_comp,
            years[0],
        )
        step_answers = []
        variable_values = []
        if len(step_questions) < 1:
            logger.error("无法解析出step questions, 答案置空")
            return ""

        for step_question, step_keyword, step_year in zip(
            step_questions, step_keywords, step_years
        ):
            if len(step_keyword) == 0:
                logger.error("关键词为空")

            background = "已知{}{}年的资料如下:\n".format(real_comp, step_year)

            matched_table_rows = AnswerGeneratorUtil.recall_pdf_tables(
                step_keyword,
                [step_year],
                pdf_table,
                min_match_number=3,
                top_k=5,
            )
            if len(matched_table_rows) == 0:
                logger.warning(
                    "无法匹配keyword {}, 尝试不设置限制".format(step_keyword)
                )
                matched_table_rows = AnswerGeneratorUtil.recall_pdf_tables(
                    step_keyword,
                    [step_year],
                    pdf_table,
                    min_match_number=2,
                    top_k=None,
                )
            if len(matched_table_rows) == 0:
                logger.error("仍然无法匹配keyword {}".format(step_keyword))
                matched_table_rows = AnswerGeneratorUtil.recall_pdf_tables(
                    step_keyword,
                    [step_year],
                    pdf_table,
                    min_match_number=0,
                    top_k=10,
                )

            table_text = AnswerGeneratorUtil.table_to_text(
                matched_table_rows,
                with_year=False,
            )
            if table_text != "":
                background += table_text

            prompt = single_question_prompt(
                ori_question,
                real_comp,
                step_year,
                background,
                step_question,
            )
            logger.debug(f"type 2 prompt for question {question_id}: {prompt}")
            step_answer = self.model(prompt)
            variable_value = AnswerGeneratorUtil.get_variable_value_from_answer(
                step_answer
            )
            if variable_value is None:
                logger.warning(
                    f"无法从答案中提取variable value, step answer: {step_answer}"
                )
                continue
            step_answers.append(step_answer)
            variable_values.append(variable_value)

        if len(step_questions) != len(variable_values):
            logger.error(
                f"step question的数量{len(step_questions)}和variable value的数量{len(variable_values)} 不一致, 答案置空"
            )
            return ""

        for name, value in zip(variable_names, variable_values):
            formula = formula.replace(name, value)
        result = None
        try:
            logger.debug(f"执行公式: {formula}")
            result = eval(formula)
        except:
            logger.error("公式函数执行失败: {}, 答案置空".format(formula))
            return ""

        if result is None:
            logger.error("公式函数执行结果为None, 答案置空")
            return ""

        answer = "".join(step_answers)
        answer += question_formula
        answer += ", 得出结果{:.2f}({:.2f}%)".format(result, result * 100)
        return answer
