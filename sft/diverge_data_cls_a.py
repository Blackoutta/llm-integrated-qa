from inferencer import VllmModel
import json
import os
from pathlib import Path


def main():
    model = VllmModel()
    prompt = """
    # 角色与任务
    你是一个专业的数据合成师，请根据我的数据示例为我合成更多数据，从而增强数据多样性.

    # 合成要求
    请生成"询问公司基本信息"相关的指令，它们必须询问关于公司的以下方面:
    - 股票简称
    - 公司名称
    - 外文名称 
    - 法定代表人
    - 注册地址
    - 办公地址
    - 公司网址网站
    - 电子信箱等

    不要出现超出这些范围的问题

    # 数据示例
    - 湖南黄金股份有限公司2019年证券代码是什么?
    - 山东阳谷华泰化工股份有限公司在2019-2021年间，法定代表人是否都相同?
    - 请提供黔源电力2019年的外文名称。
    - 2020年中远海控的法定代表人与上一年是否相同？
    - 请提供2019年春光科技的电子信箱。
    - 2019年横店集团得邦照明股份有限公司证券简称是什么?
    - 2019年天银机电外文名称是什么?
    - 2021年天津港股份有限公司法定代表人与2019年相比是否都是相同的？

    # 要求
    - 生成{num_synthesized_per_run}条
    - 语句通顺

    # 输出格式
    请按照以下markdown list格式输出, 不要输出无关内容, 不需要格式符号：
    - 问题1
    - 问题2
    - 问题3
    ...
    """
    file_dir = os.path.dirname(__file__)
    cls_dataset_dir = Path(
        Path(file_dir).parent, "resources", "dataset", "classification"
    )

    num_synthesized_total = 30
    num_synthesized_per_run = 10
    classificiation = "A"

    outputs = []
    for i in range(num_synthesized_total // num_synthesized_per_run):
        output = model.chat(
            question=prompt.format(num_synthesized_per_run=num_synthesized_per_run),
            temperature=0.4,
            max_tokens=512,
        )
        print(output)
        outputs.append(output)

    output_list = []
    for output in outputs:
        entries = output.split("-")[1:]
        entries = [entry.strip("\n").strip() for entry in entries]
        output_list.extend(entries)

    id = 1

    template = "\n        请问“{question}”是属于下面哪个类别的问题?\n        A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等.\n        B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等.\n        C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等.\n        D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重,占比等. \n        E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果.        \n        F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是XXX.\n        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.\n        "
    result_list = []
    for output in output_list:
        result_list.append(
            {
                "id": id,
                "question": output,
                "question_prompt": template.format(question=output),
                "query": classificiation,
            }
        )
        id += 1
    with open(Path(cls_dataset_dir, "temp_synthesized_data.json"), "w") as f:
        f.write(json.dumps(result_list, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
