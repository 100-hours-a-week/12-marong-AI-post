import os
import re
import torch
from typing import List, Optional

from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import PrivateAttr
from langchain.schema import BaseOutputParser
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 커스텀 래퍼
class CLOVAXLangChainWrapper(LLM):
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModelForCausalLM = PrivateAttr()
    _callbacks: CallbackManager = PrivateAttr(default_factory=lambda: CallbackManager([]))  
    _verbose: bool = PrivateAttr(default=False)
    _tags: list = PrivateAttr(default=[])
    _metadata: dict = PrivateAttr(default={})

    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../models", "hyperclovax-1.5b-instruct")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)  
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": "AI 언어모델이 하는 일은 마니또 미션을 수행한 사용자 피드에 맞춰서 적절한 mbti 수치를 바꿔주는 일이야."},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        output_ids = self._model.generate(
            **inputs, max_new_tokens=512, stop_strings=["<|endofturn|>", "<|stop|>"], tokenizer=self._tokenizer
        )
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0]

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._model

    @property
    def _llm_type(self) -> str:
        return "clovax-custom"

llm = CLOVAXLangChainWrapper()

# 출력 파서
class MBTIOutputParser(BaseOutputParser):    
    def parse(self, text: str):
        score = None
        change = "유지" 
        reason = ""      

        # 줄 단위 분할
        for line in text.splitlines():
            # score
            m_score = re.match(r"^점수\s*[:]?[\s]*(\d+)", line)
            if m_score:
                score = int(m_score.group(1))
                continue

            # change
            m_change = re.match(r"^변화\s*[:]?[\s]*(상승|하락|유지)", line)
            if m_change:
                change = m_change.group(1)
                continue

            # reason
            m_reason = re.match(r"^이유\s*[:]?[\s]*(.*)", line)
            if m_reason:
                reason = m_reason.group(1).strip()
                continue

        # 반환 시에도 항상 정의된 변수 사용
        return {
            "score": score,
            "change": change,
            "reason": reason
        }