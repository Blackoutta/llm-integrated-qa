import json
import os
from pathlib import Path
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

prompt_template = """
# 任务
请根据用户输入,为我从以下句子中提取最多3个关键词, 这些关键词是句子中最重要, 最能概括句子主题的词汇, 需要作为报表数据库查询的关键字段名被使用.

# 示例
- 输入: 请根据江化微2019年的年报,简要介绍报告期内公司主要销售客户的客户集中度情况,并结合同行业情况进行分析。
  输出: 主要销售客户集中度情况
- 输入: 能否根据2020年金宇生物技术股份有限公司的年报,给我简要介绍一下报告期内公司的社会责任工作情况？
  输出: 社会责任工作情况

# 输出格式
以markdown code block形式, 将关键词按逗号分割的形式输出, 例如:
```关键词1, 关键词2, 关键词3```

# 用户输入：
{user_input}
"""


def main():
    """
    data example:
    [
      {
      "id": 0,
      "question_prompt": "\n        请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词,你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.\n        用户输入：\n        \"能否根据2020年金宇生物技术股份有限公司的年报,给我简要介绍一下报告期内公司的社会责任工作情况？\"",
      "question": "能否根据2020年金宇生物技术股份有限公司的年报,给我简要介绍一下报告期内公司的社会责任工作情况？",
      "query": "社会责任工作情况"
      },
      {
      "id": 1,
      "question_prompt": "\n        请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词,你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.\n        用户输入：\n        \"请根据江化微2019年的年报,简要介绍报告期内公司主要销售客户的客户集中度情况,并结合同行业情况进行分析。\"",
      "question": "请根据江化微2019年的年报,简要介绍报告期内公司主要销售客户的客户集中度情况,并结合同行业情况进行分析。",
      "query": "主要销售客户集中度情况"
      },
      ...
    ]
    """

    file_dir = os.path.dirname(__file__)
    keyword_dataset_dir = Path(Path(file_dir).parent, "resources", "dataset", "keyword")

    dataset = load_dataset(
        "json",
        data_files={"raw": Path(keyword_dataset_dir, "raw_data.json").as_posix()},
    )

    def process_example(example):
        return {
            "prompt": prompt_template.format(user_input=example["question"]),
            "label": "```" + example["query"] + "```",
        }

    dataset = dataset.map(
        process_example, remove_columns=["question_prompt", "query", "id"]
    )
    dataset = dataset.shuffle(seed=1234)
    splitted = dataset["raw"].train_test_split(test_size=0.2, seed=1234)

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
