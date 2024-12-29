from ._model import InferenceModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from peft.peft_model import PeftModelForSequenceClassification
import torch


class HFModelForClassification(InferenceModel):
    def __init__(
        self,
        adapter_model_id: str = None,
        model_name: str = "Blackoutta/bert-base-chinese-sft-intention",
        cache_dir: str = "",
        id2label: dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"},
        label2id: dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
        num_labels: int = 6,
    ):
        super().__init__(model_name, cache_dir)
        self.id2label = id2label
        self.label2id = label2id
        self.adapter_model_id = adapter_model_id
        self.num_labels = num_labels

    def _chat(
        self,
        question: str,
        max_tokens: int = 1024,
        temperature=1.0,
        top_p=1.0,
        lora_name="",
    ) -> str:
        inputs = self.tokenizer(question, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            return self.id2label.get(predicted_class, "G")

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_full_path, trust_remote_code=True
        )

        if self.adapter_model_id is not None:
            adapter_model_path = os.path.join(self.cache_dir, self.adapter_model_id)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_full_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            self.model = PeftModelForSequenceClassification.from_pretrained(
                base_model, adapter_model_path
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_full_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )

    def _unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def test_cls_model():
    template = """
        请问“{question}”是属于下面哪个类别的问题?
        A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等.
        B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等.
        C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等.
        D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重,占比等. 
        E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果.        
        F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是XXX.
        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.
        """
    questions = [
        "无形资产是什么?",
        "什么是归属于母公司所有者的综合收益总额？",
        "航发动力2019年的非流动负债比率保留两位小数是多少？",
        "在上海注册的所有上市公司中，2021年货币资金最高的前3家上市公司为？金额为？",
    ]
    model = HFModelForClassification(
        adapter_model_id="Blackoutta/Qwen2.5-3B-Instruct-sft-intention-ptuningv2",
    )
    for question in questions:
        q = template.format(question=question)
        print(q)
        print(model.chat(q))
