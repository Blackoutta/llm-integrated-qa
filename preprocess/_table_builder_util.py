import re
from loguru import logger
from typing import *

alias = {
    "在职员工的数量合计": "职工总人数",
    "负债合计": "总负债",
    "资产总计": "总资产",
    "流动负债合计": "流动负债",
    "非流动负债合计": "非流动负债",
    "流动资产合计": "流动资产",
    "非流动资产合计": "非流动资产",
}


def get_unit(pdf_key, table, pages):
    unit = 1
    if len(table) == 0:
        return 1
    if len(pages) == 0:
        return 1
    page_num = table[0].strip().split("|")[1]
    for idx, page_item in enumerate(pages):
        if str(page_item["page"]) == page_num:
            last_page_lines = []
            if idx > 0:
                last_page_lines = pages[idx - 1]["text"].split("\n")[-10:]
            current_page_lines = page_item["text"].split("\n")
            search_string = None
            for line in last_page_lines + current_page_lines:
                re_unit = re.findall(r"单位\s*[:：；].{0,3}元", line) + re.findall(
                    "人民币.{0,3}元", line
                )
                if len(re_unit) != 0:
                    search_string = re_unit[0]
                    break
            if search_string is None:
                logger.debug(
                    "cannot find unit for key {} page {}".format(pdf_key, page_num)
                )
                continue
            if "百万" in search_string:
                unit = 1000000
            elif "万" in search_string:
                unit = 10000
            elif "千" in search_string:
                unit = 1000
            if unit != 1:
                break
    if unit != 1:
        logger.info("{}的单位是{}".format(pdf_key, unit))
    return unit


def table_to_tuples(pdf_key, year, table_name, table_lines, pages):
    year = year.replace("年", "")
    if table_name == "basic_info":
        r = basic_info_to_tuple(year, table_lines)
    elif table_name == "employee_info":
        r = employee_info_to_tuple(year, table_lines)
    elif table_name == "dev_info":
        r = dev_info_to_tuple(year, table_lines)
    else:
        r = fs_info_to_tuple_v1(pdf_key, table_name, year, table_lines, pages)
    results = []

    for t in r:
        t_year = t[1]
        if t_year != year:
            continue
        if t[2] in alias:
            results.append((t[0], t_year, alias[t[2]], t[3]))
        else:
            results.append((t[0], t_year, t[2], t[3]))
    return results


def is_valid_number(s):
    # 0~99
    two_digits = re.findall(r"\d\d{0,1}", s)
    if len(two_digits) == 1 and two_digits[0] == s:
        return False
    # 7-1, 1-2
    digit_broken_digit = re.findall(r"\d+-\d+", s)
    if len(digit_broken_digit) == 1 and digit_broken_digit[0] == s:
        return False
    return True


def try_multi_number(text, chosen_index=0):
    # 正则表达式匹配数字，包括小数和千位分隔符
    pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"

    # 使用正则表达式找到所有匹配的数字
    numbers = re.findall(pattern, text)

    if len(numbers) == 0:
        return text

    if len(numbers) < chosen_index + 1:
        return numbers[-1]
    # 将找到的数字按空格分割并返回
    return numbers[chosen_index]


def fs_info_to_tuple_v1(pdf_key, table_name, year, table_lines, pages):
    unit = get_unit(pdf_key, table_lines, pages)
    # print('table name and unit ', table_name, unit)
    tuples = []
    page_id = None
    for line in table_lines:
        if "page" in line:
            page_id = line.split("page")[1]
            continue
        line = line.strip("\n").split("|")
        line_text = []
        for sp in line:
            if sp == "":
                continue
            sp = re.sub("[ ,]", "", sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) == 1:
            continue
        if len(line_text) >= 2:
            row_name = line_text[0]
            row_name = re.sub("[\d \n\.．]", "", line[0])
            row_name = re.sub("（[一二三四五六七八九十]）", "", row_name)
            row_name = re.sub("\([一二三四五六七八九十]\)", "", row_name)
            row_name = re.sub("[一二三四五六七八九十][、.]", "", row_name)
            row_name = re.sub("其中：", "", row_name)
            row_name = re.sub("[加减]：", "", row_name)
            row_name = re.sub("（.*）", "", row_name)
            row_name = re.sub("\(.*\)", "", row_name)

            if row_name == "":
                continue

            row_values = []
            for value in line_text[1:]:
                if value == "" or value == "-":
                    continue
                if set(value).issubset(set("0123456789.,-")):
                    try:
                        if is_valid_number(value):
                            row_values.append("{:.2f}元".format(float(value) * unit))
                    except:
                        logger.error(
                            "Invalid value {} {} {}".format(value, pdf_key, table_name)
                        )
                        row_values.append(value + "元")
            # print(line_text)
            # print(row_values, '----')
            if len(row_values) == 1:
                # logger.warning('Invalid line(2 values) {} in {} {}'.format(line_text, table_name, year))
                tuples.append((table_name, year, row_name, row_values[0]))
            elif len(row_values) == 2:
                tuples.append((table_name, year, row_name, row_values[0]))
                tuples.append((table_name, str(int(year) - 1), row_name, row_values[1]))
            elif len(row_values) >= 3:
                tuples.append((table_name, year, row_name, row_values[1]))
                tuples.append((table_name, str(int(year) - 1), row_name, row_values[2]))
    return tuples


def basic_info_to_tuple(year, table_lines):
    tuples = []
    for line in table_lines:
        if "page" in line:
            continue
        line = line.strip("\n").split("|")
        line_text = []
        for sp in line:
            if sp == "":
                continue
            sp = sp.replace(" ", "").replace('"', "")
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 1:
            row_name = line_text[0]
            row_name = re.sub("[(（].*[）)]", "", row_name)
            row_name = re.sub("(公司|的)", "", row_name)
            # row_name = '"{}"'.format(row_name)
        if len(line_text) == 1:
            tuples.append(("basic_info", year, row_name, ""))
        elif len(line_text) == 2:
            tuples.append(("basic_info", year, row_name, line_text[1]))
        elif len(line_text) == 3:
            tuples.append(("basic_info", year, row_name, "|".join(line_text[1:])))
        elif len(line_text) >= 4:
            tuples.append(("basic_info", year, row_name, line_text[1]))
            tuples.append(("basic_info", year, line_text[2], line_text[3]))
    return tuples


def employee_info_to_tuple(year, table_lines):
    tuples = []
    for line in table_lines:
        if "page" in line:
            continue
        line = line.strip("\n").split("|")
        line_text = []
        for sp in line:
            if sp == "":
                continue
            sp = re.sub(r"[ ,]", "", sp)
            sp = re.sub(r"[(（]人[）)]", "", sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 2:
            try:
                number = float(line_text[1])
                row_name = line_text[0]
                tuples.append(("employee_info", year, row_name, line_text[1] + "人"))
            except:
                continue
    return tuples


def dev_info_to_tuple(year, table_lines):
    tuples = []
    year_index = 0
    for line in table_lines:
        if "page" in line:
            continue
        if not "研发人员" in line:
            continue
        line = line.strip("\n").split("|")
        line_text = []
        for sp in line:
            if sp == "":
                continue
            sp = re.sub("[ ,]", "", sp)
            sp = re.sub("[(（]人[）)]", "", sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 2:
            tuples.append(("dev_info", year, line_text[0], line_text[1] + "人"))
    return tuples


def test_basic_info_to_tuple():
    r = basic_info_to_tuple(
        "2019",
        [
            "公司的外文名称|Chongqing Port Co.,Ltd.",
            "公司的外文名称缩写|CQP",
            "公司的法定代表人|杨昌学",
        ],
    )
    print(r)


def test_fs_info_to_tuple():
    # data = [
    #     "项目|附注|2021年度|2020年度\n",
    #     "一、营业总收入|||14,603,100,739.78  12,825,879,050.96\n",
    #     "其中：营业收入|十、七、61||14,603,100,739.78  12,825,879,050.96\n",
    # ]
    data = [
        "项目|附注|2021 年度|2020 年度\n",
        "投资收益（损失以“－”号填列）|（四十二）|-516,494.70|43,292,122.00",
    ]

    results = fs_info_to_tuple_v1("test", "test", "2021", table_lines=data, pages=[])
    for r in results:
        print(r)


def test_employee_info_to_tuple():
    data = [
        "page|48" "行政人员|231" "合计|3,395",
        "教育程度|",
        "教育程度类别|数量（人）",
        "博士|14",
    ]
    results = employee_info_to_tuple(year="2021", table_lines=data)
    for r in results:
        print(r)
