from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
import torch
from loguru import logger


class InferenceModel(ABC):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: str = "",
    ):
        self.model_name = model_name
        self.cache_dir = (
            os.path.join(os.environ["HOME"], ".cache/modelscope/hub")
            if cache_dir == ""
            else cache_dir
        )
        self.model_full_path = os.path.join(self.cache_dir, self.model_name)
        self.loaded = False

    @abstractmethod
    def _chat(
        self,
        question: str,
        max_tokens: int = 2048,
        temperature=0.01,
        top_p=0.8,
        lora_name="",
    ) -> str:
        pass

    def chat(
        self,
        question: str,
        max_tokens: int = 2048,
        temperature=0.01,
        top_p=0.8,
        lora_name="",
    ) -> str:
        if not self.loaded:
            self._load()
            self.loaded = True
        return self._chat(question, max_tokens, temperature, top_p, lora_name)

    @abstractmethod
    def _load(self):
        pass

    def unload(self):
        if self.loaded:
            self._unload()

    @abstractmethod
    def _unload(self):
        pass

    def __call__(self, *args, **kwds):
        return self.chat(*args, **kwds)


class MockModel(InferenceModel):
    def __init__(self):
        super().__init__("", "")

    def _load(self):
        pass

    def _unload(self):
        pass

    def _chat(
        self, question: str, max_tokens: int = 1024, temperature=1.0, top_p=1.0
    ) -> str:
        return "Mock answer"


class VllmModel(InferenceModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: str = "",
        lora_adapters: dict = {},
    ):
        super().__init__(model_name, cache_dir)
        self.lora_adapters = lora_adapters
        self.lora_request_map = {}
        self.llm: LLM = None

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_full_path, trust_remote_code=True
        )
        id = 1
        if len(self.lora_adapters) > 0:
            for lora_desc, lora_path in self.lora_adapters.items():
                self.lora_request_map[lora_desc] = {"id": id, "path": lora_path}
                id += 1

            self.llm = LLM(
                model=self.model_full_path,
                gpu_memory_utilization=0.9,
                max_model_len=16384,
                enable_lora=True,
            )
            print(self.lora_request_map)
            return

        self.llm = LLM(
            model=self.model_full_path,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
        )

    def _unload(self):
        del self.llm
        del self.tokenizer
        torch.cuda.empty_cache()

    def _chat(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature=0.01,
        top_p=0.8,
        lora_name="",
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        sp = (
            SamplingParams(
                stop=["<|im_end|>", "``` "],
                top_k=20,
                top_p=top_p,
                repetition_penalty=1.05,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        lora_config: dict = None
        if lora_name != "":
            lora_config = self.lora_request_map.get(lora_name, None)

        if lora_config is None:
            # do normal generation
            outputs = self.llm.chat(
                use_tqdm=False,
                messages=messages,
                sampling_params=sp,
            )
            return outputs[0].outputs[0].text.strip("```").strip()

        # do lora generation
        input_text = self.llm.get_tokenizer().apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.llm.generate(
            input_text,
            sampling_params=sp,
            use_tqdm=False,
            lora_request=LoRARequest(
                lora_name=lora_name,
                lora_path=lora_config["path"],
                lora_int_id=lora_config["id"],
            ),
        )
        return outputs[0].outputs[0].text.strip("```").strip()
