from inferencer import *
from evaluator import *
import os
from pathlib import Path
from datetime import datetime
from functools import partial
from dotenv import load_dotenv

file_dir = os.path.dirname(__file__)


def main():
    """
    运行前需设置环境变量
    - MODEL_CACHE_DIR: 模型缓存目录
    """

    load_dotenv()
    model_cache_dir = os.environ.get("MODEL_CACHE_DIR")

    sdn = get_sample_dir_name()

    # 分类模型
    cls_model = HFModelForClassification(
        model_name="Blackoutta/bert-base-chinese-sft-intention",
    )

    # 关键词及nl2sql模型
    lora_model = VllmModel(
        lora_adapters={
            "keywords": os.path.join(
                model_cache_dir, "Blackoutta/Qwen2.5-3B-Instruct-sft-keyword-lora"
            ),
            "nl2sql": os.path.join(
                model_cache_dir, "Blackoutta/Qwen2.5-3B-Instruct-sft-nl2sql-lora"
            ),
        },
    )

    # 通用模型
    generic_model = VllmModel()

    inferenced_dir = Path(file_dir, "resources/inferenced", sdn)

    # 评估器
    evaluator = Evaluator(
        metrics_input=Path(file_dir, "resources/metrics"),
        answer_input=inferenced_dir,
        eval_output=Path(file_dir, "resources/evaluated", sdn),
    )

    # 推理器
    inferencer = Inferencer(
        cls_model=cls_model,
        default_model=lora_model,
        generic_model=generic_model,
        ctx_dir=Path(file_dir, "resources", "processed_data"),
        inference_dir=inferenced_dir,
    )
    questions = evaluator.load_questions()
    questions = [q.model_dump() for q in questions]

    # TODO for test and debug purposes
    # questions = [questions[47]]
    # questions = questions[:20]

    # 问题分类推理
    inferencer.do_classification(questions=questions, persist=True)

    # 关键词推理
    inferencer.do_keywords_generation(
        questions=questions, persist=True, lora_name="keywords"
    )

    # NL2SQl推理
    inferencer.do_sql_generation(
        questions=questions, persist=True, lora_name="nl2sql", unload_on_done=True
    )

    # 生成答案
    inferencer.do_answer_generation(questions=questions, persist=True)

    # 评估分数
    evaluator.do_evaluation(persist=True)


def get_sample_dir_name():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


if __name__ == "__main__":
    main()
