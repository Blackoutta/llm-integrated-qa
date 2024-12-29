import json
import os
import random
from pathlib import Path


def main():
    """
      data example:
      [
        {
            "id": 0,
            "question_prompt": "\n        请问“能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？”是属于下面哪个类别的问题?\n        A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等.\n        B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等.\n        C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等.\n        D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重,占比等. \n        E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果.        \n        F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是XXX.\n        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.\n        ",
            "question": "能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？",
            "query": "F"
        },
        {
            "id": 1,
            "question_prompt": "\n        请问“请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。”是属于下面哪个类别的问题?\n        A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等.\n        B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等.\n        C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等.\n        D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重,占比等. \n        E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果.        \n        F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是XXX.\n        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.\n        ",
            "question": "请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。",
            "query": "F"
        }
    ...
    ]

    """
    file_dir = os.path.dirname(__file__)
    cls_dataset_dir = Path(
        Path(file_dir).parent, "resources", "dataset", "classification"
    )

    with open(
        Path(
            cls_dataset_dir,
            "diverged_data.json",
        ),
        "r",
    ) as f:
        data = json.load(f)

    # data group by query
    data_grouped = {}
    for item in data:
        item["query"] = item["query"].strip().replace("。", "")
        query = item["query"]
        if query not in data_grouped:
            data_grouped[query] = []
        data_grouped[query].append(item)

    # randomly pick 10% from each group and make a new group
    data_grouped_new = {}
    for query, items in data_grouped.items():
        sample_ratio = 0.1
        data_grouped_new[query] = random.sample(items, int(len(items) * sample_ratio))

    # remove elements in data group new from data group
    for query, items in data_grouped_new.items():
        for item in items:
            if item in data_grouped[query]:
                data_grouped[query].remove(item)

    # count data group
    print("data group count:")
    for query, items in data_grouped.items():
        print(f"query: {query}, count: {len(items)}")

    # count data group new
    print("data group new count:")
    for query, items in data_grouped_new.items():
        print(f"query: {query}, count: {len(items)}")

    # flatten data group and data group new to list
    data_grouped_list = [
        {
            "class": item["query"],
            "question": item["question"],
            "prompt": item["question_prompt"],
        }
        for items in data_grouped.values()
        for item in items
    ]
    data_grouped_list.sort(key=lambda x: x["class"])

    data_grouped_new_list = [
        {
            "class": item["query"],
            "question": item["question"],
            "prompt": item["question_prompt"],
        }
        for items in data_grouped_new.values()
        for item in items
    ]
    data_grouped_new_list.sort(key=lambda x: x["class"])

    # write data grouped list as train.jsonl
    with open(Path(cls_dataset_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in data_grouped_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write data grouped new list as test.jsonl
    with open(Path(cls_dataset_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for item in data_grouped_new_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
