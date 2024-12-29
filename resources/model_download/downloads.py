# 模型下载
from modelscope import snapshot_download

# 默认下载到: ~/.cache/modelscope/hub/{repo}/{model_name}

# base models
snapshot_download("Qwen/Qwen2.5-3B-Instruct")

snapshot_download("Jerry0/text2vec-base-chinese")

snapshot_download("tiansz/bert-base-chinese")

# fine-tuned models
snapshot_download("Blackoutta/bert-base-chinese-sft-intention")

snapshot_download("Blackoutta/Qwen2.5-3B-Instruct-sft-keyword-lora")

snapshot_download("Blackoutta/Qwen2.5-3B-Instruct-sft-nl2sql-lora")
