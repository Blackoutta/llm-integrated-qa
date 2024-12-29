from ._answer_generator import AnswerGenerator
from dataloader import DataLoader
from ._model import InferenceModel
from ._answer_generator_util import AnswerGeneratorUtil
import re
from loguru import logger
from ._prompt import sql_correction_prompt, find_synonyms_prompt, general_qa_prompt
from typing import List


class AnswerGeneratorSql(AnswerGenerator):
    def __init__(self, dataloader: DataLoader, model: InferenceModel):
        super().__init__(dataloader, model)

    def generate_answer(self, question_id, question, question_type) -> str:
        sql_dict = self.dataloader.load_nl2sql_map()
        sql = sql_dict[question_id]
        existing_fields = list(
            self.dataloader.load_company_table(remove_column_prefix=True).columns
        )

        if sql is None:
            return ""

        ori_question = AnswerGenerator.cleanup_question(question)

        sql = AnswerGeneratorSql.correct_sql_number(sql, ori_question)
        sql_ctx, exec_err = self.dataloader.exec_sql(sql)

        if exec_err is None:
            return self._gen_answer_with_model(ori_question, sql_ctx, question_type)

        logger.warning(
            f"执行sql错误: {exec_err}, sql: {sql}, question_id: {question_id}"
        )

        # sql错误尝试修复一次
        if "no such column" in exec_err:
            wrong_column = exec_err.replace("no such column:", "").strip()
            logger.info(f"尝试修复no such column错误, 目标字段: {wrong_column}")

            sql = self.correct_sql_field_llm(sql, existing_fields, wrong_column)
            sql_ctx, exec_err = self.dataloader.exec_sql(sql)
            if exec_err is not None:
                logger.error(
                    f"重试no such column后, 依然错误, sql: [{sql}]\n错误信息: {exec_err}, 答案置空"
                )
                return ""
            logger.info(f"no such column重试成功! sql: [{sql}], 答案: {sql_ctx}")
            return self._gen_answer_with_model(ori_question, sql_ctx, question_type)

        logger.info(f"尝试修复sql错误...")
        logger.debug("模型纠正前sql: {}".format(sql.replace("<>", "")))
        corrected_sqls = self.model(
            sql_correction_prompt(existing_fields, sql, exec_err)
        )
        logger.debug("模型纠正后sql: {}".format(corrected_sqls.replace("<>", "")))

        corrected_sql_groups = re.findall("```sql([\s\S]+)```", corrected_sqls)
        if len(corrected_sql_groups) < 1:
            logger.error("无法从纠正后sql中提取出sql语句, 答案置空")
            return ""

        corrected_sql = corrected_sql_groups[0].replace("\n", "").strip()
        sql_ctx, exec_err = self.dataloader.exec_sql(corrected_sql)
        if exec_err is not None:
            logger.error(
                "重试纠正后的SQL依然错误, q_id: {question_id},  sql: [{}]\n错误信息: {}, 答案置空".format(
                    question_id, corrected_sql, exec_err
                )
            )
            return ""
        return self._gen_answer_with_model(ori_question, sql_ctx, question_type)

    def _gen_answer_with_model(
        self, question: str, sql_ctx: dict, question_type: str
    ) -> str:
        if "第" in question and "高" in question:
            sql_ctx["result"] = sql_ctx["result"].split("\n")[0]
        prompt = general_qa_prompt(question=question, ctx=sql_ctx)
        logger.debug(f"prompt for type {question_type}: {prompt}")
        return self.model.chat(question=prompt)

    @staticmethod
    def correct_sql_number(sql, question):
        new_sql = sql
        fields, sql_numbers = AnswerGeneratorSql.get_field_number(sql)
        q_numbers = AnswerGeneratorSql.get_number_from_question(question)
        for sql_number in sql_numbers:
            if (
                len(sql_number) > 2
                and sql_number not in q_numbers
                and len(q_numbers) == 1
            ):
                logger.info("文本数字纠正前sql：{}".format(new_sql))
                new_sql = new_sql.replace(sql_number, q_numbers[0])
                logger.info("文本数字纠正后sql：{}".format(new_sql))
        return new_sql

    @staticmethod
    def get_field_number(sql):
        sql_words = sql.split(" ")
        fields = []
        numbers = []
        pre_word = ""
        for word in sql_words:
            if word == "" or word in ["(", ")"]:
                continue
            if word.startswith("("):
                word = word[1:]
            # 只检查条件字段
            if pre_word in ["and", "or", "by", "where"] and re.match(
                r"^[\u4E00-\u9FA5]+$", word
            ):
                fields.append(word)
            elif (
                pre_word in ["<", ">"] and re.match(r"^[0-9]+$", word) and len(word) > 2
            ):
                numbers.append(word)
            pre_word = word
        return fields, numbers

    @staticmethod
    def extract_zh_field_names(sql: str) -> List[str]:
        """
        从一段sql中解析出中文字段名
        """
        pattern = r"[^\w](\b[\u4e00-\u9fa5]+\b)"

        # 使用findall方法查找所有匹配的字段名
        field_names = re.findall(pattern, sql)
        return field_names

    @staticmethod
    def get_number_from_question(question):
        unit_dic = {
            "十万": 100000,
            "百万": 1000000,
            "千万": 10000000,
            "十亿": 1000000000,
            "百亿": 10000000000,
            "千亿": 100000000000,
            "百": 100,
            "千": 1000,
            "万": 10000,
            "亿": 100000000,
        }
        num_dic = {
            "一": 1,
            "二": 2,
            "两": 2,
            "俩": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }

        numbers = re.findall(
            "([一二三四五六七八九十两1234567890]+个?(十万|百万|千万|十亿|百亿|千亿|百|千|万|亿|))",
            question,
        )
        number_list = []
        for number in numbers:
            digit_num = number[0].replace("个", "")
            if len(number[1]) > 0:
                digit_num = digit_num.replace(number[1], "")
            if len(digit_num) > 0 and digit_num[-1] in ["十", "百", "千", "万"]:
                unit = digit_num[-1] + number[1]
                digit_num = digit_num[:-1]
            else:
                unit = number[1]
            # 太小的纯数字和年份不作检查
            if unit == "" and (
                len(digit_num) < 3 or (len(digit_num) == 4 and digit_num[:2] == "20")
            ):
                continue
            # 纯数字，不带单位
            elif unit == "" and re.match("^[0-9]+$", digit_num):
                number_list.append(digit_num)
            # 十亿、百亿类直接是单位
            elif digit_num == "" and len(unit) == 2 and unit in unit_dic.keys():
                number_list.append(str(unit_dic.get(unit)))
            # 带单位
            elif unit in unit_dic.keys():
                digit_num = digit_num.replace(unit, "")
                if digit_num in num_dic.keys():
                    digit_num = num_dic.get(digit_num)
                    number_list.append(str(digit_num * unit_dic.get(unit)))
                elif re.match("^[0-9]+$", digit_num):
                    number_list.append(str(int(digit_num) * unit_dic.get(unit)))
        return number_list

    def correct_sql_field_llm(self, sql, existing_fields, wrong_column):
        """
        用大模型来根据已知字段纠正sql中错误的字段
        """
        new_sql = sql
        synonyms = self.find_synonyms_llm(wrong_column, existing_fields)
        if len(synonyms) > 0:
            logger.debug("文本字段纠正前sql: {}".format(new_sql))
            new_sql = new_sql.replace(wrong_column, synonyms)
            logger.debug("文本字段纠正后sql: {}".format(new_sql))
        return new_sql

    def find_synonyms_llm(self, word, word_lsit):
        try:
            answer = self.model(find_synonyms_prompt(word_lsit, word))
            logger.debug("同义词推理结果：{}".format(answer.replace("<>", "")))
        except Exception as e:
            logger.warning(
                "模型查询同义词字段失败：{}".format(str(e).replace("<>", ""))
            )
        return answer
