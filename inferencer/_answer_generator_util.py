from typing import *
import re
import pandas as pd
from difflib import SequenceMatcher
from loguru import logger


class AnswerGeneratorUtil:
    @staticmethod
    def extract_years(question) -> List[str]:
        years = re.findall("\d{4}", question)

        if len(years) == 1:
            if (
                re.search("(([上前去]的?[1一]|[上去])年|[1一]年(前|之前))", question)
                and "上上年" not in question
            ):
                last_year = int(years[0]) - 1
                years.append(str(last_year))
            if re.search("((前|上上)年|[2两]年(前|之前))", question):
                last_last_year = int(years[0]) - 2
                years.append(str(last_last_year))
            if re.search("[上前去]的?[两2]年", question):
                last_year = int(years[0]) - 1
                last_last_year = int(years[0]) - 2
                years.append(str(last_year))
                years.append(str(last_last_year))

            if re.search("([后下]的?[1一]年|[1一]年(后|之后|过后))", question):
                next_year = int(years[0]) + 1
                years.append(str(next_year))
            if re.search("[2两]年(后|之后|过后)", question):
                next_next_year = int(years[0]) + 2
                years.append(str(next_next_year))
            if re.search("(后|接下来|下)的?[两2]年", question):
                next_year = int(years[0]) + 1
                next_next_year = years[0] + 2
                years.append(str(next_year))
                years.append(str(next_next_year))

        if len(years) == 2:
            if re.search("\d{4}年?[到\-至]\d{4}年?", question):
                year0 = int(years[0])
                year1 = int(years[1])
                for year in range(min(year0, year1) + 1, max(year0, year1)):
                    years.append(str(year))

        return years

    @staticmethod
    def get_matching_substrs(a, b):
        return "".join(set(a).intersection(b))

    @staticmethod
    def recall_pdf_tables(
        keywords,
        years,
        tables,  # [(table_name, row_year, row_name, row_value)]
        valid_tables=None,
        invalid_tables=None,
        min_match_number=3,
        top_k=None,
    ):
        valid_keywords = keywords

        matched_lines = []
        for table_row in tables:
            table_name, row_year, row_name, row_value = table_row
            row_name = row_name.replace('"', "")
            if row_year not in years:
                continue

            if valid_tables is not None and table_name not in valid_tables:
                continue

            if invalid_tables is not None and table_name in invalid_tables:
                continue

            # find exact match, only return this row
            if row_name == valid_keywords:
                matched_lines = [(table_row, len(row_name))]
                break

            tot_match_size = 0
            matches = SequenceMatcher(
                isjunk=None,
                a=valid_keywords,
                b=row_name,
                autojunk=False,
            )
            for match in matches.get_matching_blocks():
                inter_text = valid_keywords[match.a : match.a + match.size]
                tot_match_size += match.size
            if tot_match_size >= min_match_number or row_name in valid_keywords:
                matched_lines.append([table_row, tot_match_size])

        matched_lines = sorted(matched_lines, key=lambda x: x[1], reverse=True)
        matched_lines = [t[0] for t in matched_lines]
        if top_k is not None and len(matched_lines) > top_k:
            matched_lines = matched_lines[:top_k]
        return matched_lines

    @staticmethod
    def table_to_dataframe(table_rows):
        df = pd.DataFrame(
            table_rows, columns=["table_name", "row_year", "row_name", "row_value"]
        )
        df["row_year"] = pd.to_numeric(df["row_year"])
        df.drop_duplicates(inplace=True)
        df.sort_values(by=["row_name", "row_year"], inplace=True)
        return df

    @staticmethod
    def table_to_text(table_rows, with_year=True):
        text_lines = []
        for row in table_rows:
            table_name, row_year, row_name, row_value = row
            if pd.isna(row_value):
                continue

            if table_name == "basic_info":
                row_value = '"{}"'.format(row_value)
            else:
                row_name = '"{}"'.format(row_name)

            if not with_year:
                row_year = ""
            else:
                row_year += "年的"
            if row_value in ["相同", "不相同且不同"]:
                line = "{}的{}{},".format(row_year, row_name, row_value)
            else:
                if table_name == "employee_info":
                    line = "{}{}有{},".format(row_year, row_name, row_value)
                else:
                    line = "{}{}是{},".format(row_year, row_name, row_value)
            if line not in text_lines:
                text_lines.append(line)
        return "".join(text_lines)

    @staticmethod
    def add_text_compare_in_table(table_rows):
        df = AnswerGeneratorUtil.table_to_dataframe(table_rows)
        added_rows = []
        for idx, (index, row) in enumerate(df.iterrows()):
            if idx == 0:
                continue

            last_row = df.iloc[idx - 1]
            if last_row["row_name"] == row["row_name"]:
                last_values = AnswerGeneratorUtil.find_numbers(last_row["row_value"])
                current_values = AnswerGeneratorUtil.find_numbers(row["row_value"])
                if len(last_values) == 0 and len(current_values) == 0:
                    if row["row_value"] != last_row["row_value"]:
                        row_value = "不相同且不同"
                    else:
                        row_value = "相同"
                    added_rows.append(
                        [
                            row["table_name"],
                            "{}与{}相比".format(row["row_year"], last_row["row_year"]),
                            row["row_name"],
                            row_value,
                        ]
                    )
        merged_rows = table_rows + added_rows
        return merged_rows

    @staticmethod
    def find_numbers(s: Union[str, int, float]):
        if isinstance(s, (int, float)):
            return [s]
        numbers = re.findall("[-\d,\.]+", s)
        float_numbers = []
        for number in numbers:
            try:
                float_numbers.append(float(number))
            except:
                pass
        return float_numbers

    @staticmethod
    def anoy_question_xx(question, real_company, years):
        question_new = question
        question_new = question_new.replace(real_company, "XX公司")
        for year in years:
            question_new = question_new.replace(year, "XXXX")

        return question_new

    @staticmethod
    def parse_keyword_from_answer(anoy_question, answer):
        key_words = set()
        key_word_list = answer.split("\n")
        for key_word in key_word_list:
            key_word = key_word.replace(" ", "")
            # key_word = re.sub('年报|报告|是否', '', key_word)
            if (
                key_word.endswith("公司") and not key_word.endswith("股公司")
            ) or re.search(
                r"(年报|财务报告|是否|最高|最低|相同|一样|相等|在的?时候|财务数据|详细数据|单位为|年$)",
                key_word,
            ):
                continue
            if key_word.startswith("关键词"):
                key_word = re.sub("关键词[1-9][:|：]", "", key_word)
                if key_word in ["金额", "单位", "数据"]:
                    continue
                if key_word in anoy_question and len(key_word) > 1:
                    key_words.add(key_word)
        return list(key_words)

    @staticmethod
    def merge_idx(indexes, total_len, prefix=0, suffix=1):
        merged_idx = []
        for index in indexes:
            start = max(0, index - prefix)
            end = min(total_len, index + suffix + 1)
            merged_idx.extend([i for i in range(start, end)])
        merged_idx = sorted(list(set(merged_idx)))

        block_idxes = []

        if len(merged_idx) == 0:
            return block_idxes

        current_block_idxes = [merged_idx[0]]
        for i in range(1, len(merged_idx)):
            if merged_idx[i] - merged_idx[i - 1] > 1:
                block_idxes.append(current_block_idxes)
                current_block_idxes = [merged_idx[i]]
            else:
                current_block_idxes.append(merged_idx[i])
        if len(current_block_idxes) > 0:
            block_idxes.append(current_block_idxes)

        return block_idxes

    @staticmethod
    def rewrite_text_block(text):
        for word in ["是", "否", "适用", "不适用"]:
            text = text.replace("□{}".format(word), "")
        return text

    @staticmethod
    def is_type2_growth_rate(question):
        # 问题不包含年份
        if len(re.findall("\d{4}", question)) == 0:
            return False
        if "增长率" in question:
            return True
        return False

    def add_growth_rate_in_table(table_rows):
        df = AnswerGeneratorUtil.table_to_dataframe(table_rows)
        added_rows = []
        for idx, (index, row) in enumerate(df.iterrows()):
            last_row = df.iloc[idx - 1]
            if (
                last_row["row_name"] == row["row_name"]
                and last_row["row_year"] == row["row_year"] - 1
            ):
                last_values = AnswerGeneratorUtil.find_numbers(last_row["row_value"])
                current_values = AnswerGeneratorUtil.find_numbers(row["row_value"])
                if len(last_values) > 0 and len(current_values) > 0:
                    if last_values[0] != 0:
                        growth_rate = (
                            (current_values[0] - last_values[0]) / last_values[0] * 100
                        )
                        added_rows.append(
                            [
                                row["table_name"],
                                str(row["row_year"]),
                                row["row_name"] + "增长率",
                                "{:.2f}%".format(growth_rate),
                            ]
                        )
        merged_rows = table_rows + added_rows
        return merged_rows

    @staticmethod
    def get_formulas():
        formulas = [
            "研发经费与利润=研发费用/净利润",
            "研发经费与营业收入=研发费用/营业收入",
            "研发人员占职工=研发人员的数量/在职员工的数量合计",
            "研发人员占总职工=研发人员的数量/在职员工的数量合计",
            "研发人员在职工=研发人员的数量/在职员工的数量合计",
            "研发人员所占=研发人员的数量/在职员工的数量合计",
            "流动比率=流动资产合计/流动负债合计",
            "速动比率=(流动资产合计-存货)/流动负债合计",
            "硕士及以上人员占职工=(硕士研究生+博士)/在职员工的数量合计",
            "硕士及以上学历的员工占职工=(硕士研究生+博士)/在职员工的数量合计",
            "硕士及以上学历人员占职工=(硕士研究生+博士)/在职员工的数量合计",
            "研发经费占费用=研发费用/(销售费用+财务费用+管理费用+研发费用)",
            "研发经费在总费用=研发费用/(销售费用+财务费用+管理费用+研发费用)",
            "研发经费占总费用=研发费用/(销售费用+财务费用+管理费用+研发费用)",
            "营业利润率=营业利润/营业收入",
            "资产负债比率=负债合计/资产总计",
            "现金比率=货币资金/流动负债合计",
            "非流动负债比率=非流动负债合计/总负债",
            "流动负债比率=流动负债合计/总负债",
            "流动负债的比率=流动负债合计/总负债",
            "净资产收益率=净利润/净资产",
            "净利润率=净利润/营业收入",
            "营业成本率=营业成本/营业收入",
            "管理费用率=管理费用/营业收入",
            "财务费用率=财务费用/营业收入",
            "毛利率=(营业收入-营业成本)/营业收入",
            "三费比重=(销售费用+管理费用+财务费用)/营业收入",
            "三费（销售费用、管理费用和财务费用）占比=(销售费用+管理费用+财务费用)/营业收入",
            "投资收益占营业收入=投资收益/营业收入",
        ]
        formulas = [t.split("=") for t in formulas]
        return formulas

    @staticmethod
    def is_type2_formula(question):
        if len(re.findall("\d{4}", question)) == 0:
            return False
        formulas = AnswerGeneratorUtil.get_formulas()
        for k, v in formulas:
            if k in question:
                return True
        return False

    @staticmethod
    def growth_formula():
        formulas = [
            "销售费用增长率=(销售费用-上年销售费用)/上年销售费用",
            "财务费用增长率=(财务费用-上年财务费用)/上年财务费用",
            "管理费用增长率=(管理费用-上年管理费用)/上年管理费用",
            "研发费用增长率=(研发费用-上年研发费用)/上年研发费用",
            "负债合计增长率=(负债合计-上年负债合计)上年总负债",
            "总负债增长率=(总负债-上年总负债)/上年总负债",
            "流动负债增长率=(流动负债-上年流动负债)/上年流动负债",
            "货币资金增长率=(货币资金-上年货币资金)/上年货币资金",
            "固定资产增长率=(固定资产-上年固定资产)/上年固定资产",
            "无形资产增长率=(无形资产-上年无形资产)/上年无形资产",
            "资产总计增长率=(资产总计-上年资产总计)/上年资产总计",
            "投资收益增长率=(投资收益-上年投资收益)/上年投资收益",
            "总资产增长率=(资产总额-上年资产总额)/上年资产总额",
            "营业收入增长率=(营业收入-上年营业收入]/上年营业收入",
            "营业利润增长率=(营业利润-上年营业利润)/上年营业利润",
            "净利润增长率=(净利润-上年净利润)/上年净利润",
            "现金及现金等价物增长率=(现金及现金等价物-上年现金及现金等价物)/上年现金及现金等价物",
        ]
        formulas = [t.split("=") for t in formulas]
        return formulas

    @staticmethod
    def get_keywords_of_formula(value):
        keywords = re.split("[(+-/)]", value)
        keywords = [t for t in keywords if len(t) > 0]
        return keywords

    @staticmethod
    def get_step_questions(question, keywords, real_comp, year):
        new_question = question
        step_questions = []
        question_keywords = []
        variable_names = []
        step_years = []
        formula = None
        question_formula = None

        if "增长率" in question:
            if keywords == "增长率":
                keywords = new_question
            question_keywords = [keywords.replace("增长率", "")] * 2 + [keywords]
            variable_names = ["A", "B", "C"]
            formula = "(A-B)/B"
            question_formula = "根据公式，=(-上年)/上年"
            for formula_key, formula_value in AnswerGeneratorUtil.growth_formula():
                if formula_key in new_question.replace("的", ""):
                    question_formula = "根据公式，{}={},".format(
                        formula_key, formula_value
                    )
            step_years = [year, str(int(year) - 1), year]
            step_questions.append(new_question.replace("增长率", ""))
            step_questions.append(
                new_question.replace("增长率", "").replace(year, str(int(year) - 1))
            )
            step_questions.append(new_question)
        else:
            formulas = AnswerGeneratorUtil.get_formulas()
            for k, v in formulas:
                if k in new_question:
                    variable_names = AnswerGeneratorUtil.get_keywords_of_formula(v)
                    formula = v
                    for name in variable_names:
                        if (
                            "人数" in question
                            or "数量" in question
                            or "人员" in question
                        ):
                            step_questions.append(
                                "{}年{}{}有多少人?如果已知信息没有提供, 你应该回答为0人。".format(
                                    year, real_comp, name
                                )
                            )
                        else:
                            step_questions.append(
                                "{}年{}的{}是多少元?".format(year, real_comp, name)
                            )

                        question_keywords.append(name)
                        step_years.append(year)
                    question_formula = "根据公式，{}={}".format(k, v)
                    break
        return (
            step_questions,
            question_keywords,
            variable_names,
            step_years,
            formula,
            question_formula,
        )

    @staticmethod
    def get_variable_value_from_answer(answer):
        numbers = re.findall(r"[+\-\d\.]*", answer)
        numbers = [
            t for t in numbers if t not in ["2018", "2019", "2020", "2021", "2022"]
        ]
        numbers = sorted(numbers, key=lambda x: len(x), reverse=True)
        if len(numbers) >= 1:
            return numbers[0]
        else:
            return None
