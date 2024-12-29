import json
import os
from pathlib import Path
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import re

prompt_template = """
# 任务
你是一名Mysql数据库开发人员, 你精通Mysql数据库的sql语句编写, 你需要根据已知的表名、字段名和用户输入来编写sql代码.

# 上下文
## 已知表名
- company_table

## 已知字段名
- 公司全称
- 年份
- 经营活动现金流入小计
- 公司的中文简称
- 固定资产
- 应交税费
- 应付职工薪酬
- 未分配利润
- 负债合计
- 电子信箱
- 资产总计
- 无形资产
- 货币资金
- 资本公积
- 利息收入
- 营业收入
- 营业外支出
- 盈余公积
- 营业利润
- 营业外收入
- 所得税费用
- 其他收益
- 现金及现金等价物净增加额
- 净利润
- 其他应收款
- 营业成本
- 综合收益总额
- 流动资产合计
- 应收账款
- 预付款项
- 其他应付款
- 非流动资产合计
- 基本每股收益
- 购买商品
- 接受劳务支付的现金
- 应付账款
- 流动负债合计
- 利润总额
- 管理费用
- 其他流动资产
- 递延所得税资产
- 财务费用
- 营业总收入
- 非流动负债合计
- 存货
- 分配股利
- 利润或偿付利息支付的现金
- 稀释每股收益
- 所有者权益合计
- 营业总成本
- 销售费用
- 负债和所有者权益总计
- 持续经营净利润
- 信用减值损失
- 财务人员
- 销售人员
- 投资收益
- 行政人员
- 技术人员
- 利息费用
- 生产人员
- 研发费用
- 资产减值损失
- 递延收益
- 其他非流动资产
- 短期借款
- 在职员工的数量合计

# 要求
- sql代码中的字段名必须是已知字段名,不得新增字段名

# 示例
- 输入:在上海注册的上市公司中,2019年谁的负债合计最高?金额是?
  输出: ```select 公司全称, 负债合计 from company_table where 注册地址 LIKE '%上海%' 'and 年份 = '2019' order by 负债合计 desc limit 1```
- 输入:2019年负债合计最高的十家公司分别是?
  输出: ```select 公司全称 from company_table where 年份 = '2019' order by 负债合计 desc limit 10```
- 输入:在上海注册的上市公司中,2019年负债合计最多的十家公司分别是,负债合计金额分别是?
  输出: ```select 公司全称, 负债合计 from company_table where 注册地址 LIKE '%上海%' and 年份 = '2019' order by 负债合计 desc limit 10```
- 输入:注册地点在深圳市的公司中,2021年负债合计超过了五千万的公司有几家?
  输出: ```select count(1) from company_table where 年份 = '2021' and 注册地址 like '%深圳市%' and 负债合计 is not null and 负债合计 > 50000000 ```
- 输入:注册地点在四川的公司中,2019年平均的利润总额是多少?
  输出:```select avg(利润总额) from company_table where 年份 = '2019' and 注册地址 like '%四川%' and 利润总额 is not null```
- 输入:2021年注册地在上海的上市公司中,一共有多少销售人员?
  输出: ```select sum(销售人员) from company_table where 年份 = '2021' and 注册地址 like '%上海%' and 销售人员 is not null```

# 输出格式
以markdown的code block格式输出sql代码,例如:
```select sum(销售人员) from company_table where 年份 = '2021' and 注册地址 like '%上海%' and 销售人员 is not null```

# 用户输入
{user_input}
"""


def main():
    """
    data example:
    [
      {
      "question": "x",
      "answer": "y"
      },
      ...
    ]
    """

    file_dir = os.path.dirname(__file__)
    keyword_dataset_dir = Path(Path(file_dir).parent, "resources", "dataset", "nl2sql")

    dataset = load_dataset(
        "json",
        data_files={"raw": Path(keyword_dataset_dir, "raw_data.jsonl").as_posix()},
    )

    def process_example(example):
        match = re.search(
            r"请根据以下用户输入，输出sql代码。\n用户输入：(.*)", example["question"]
        )
        if match:
            user_input = match.group(1)
        else:
            user_input = ""

        pattern = r"```sql\n(.*?)\n```"
        match = re.search(pattern, example["answer"])
        if match:
            label = (
                "```"
                + match.group(0)
                .replace("\n", "")
                .replace("```sql", "")
                .replace("```", "")
                .strip()
                + "```"
            )
        else:
            label = ""

        return {
            "prompt": prompt_template.format(user_input=user_input),
            "label": label,
            "question": user_input,
        }

    dataset = dataset.map(process_example, remove_columns=["question", "answer"])
    dataset = dataset.shuffle(seed=1234)
    splitted = dataset["raw"].train_test_split(test_size=0.1, seed=1234)

    # save train and test set to jsonl
    for split, split_dataset in splitted.items():
        ds: Dataset = split_dataset
        ds.to_json(
            Path(keyword_dataset_dir, f"{split}.jsonl").as_posix(),
            orient="records",
            lines=True,
            force_ascii=False,
        )


if __name__ == "__main__":
    main()
