import os
import re
import torch
from typing import List, Optional

from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


# 모델 로딩
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir,  "../models", "hyperclovax-1.5b-instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_path, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto"
)

# 커스텀 래퍼
class CLOVAXLangChainWrapper(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": "AI 언어모델이 하는 일은 사용자 피드에 맞춰서 적절한 mbti 수치를 바꿔주는 일이다."},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            stop_strings=["<|endofturn|>", "<|stop|>"],
            tokenizer=tokenizer
        )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0]

    @property
    def _llm_type(self) -> str:
        return "clovax-custom"

llm = CLOVAXLangChainWrapper()

#  출력 파서
class MBTIOutputParser(BaseOutputParser):
    def parse(self, text: str):
        score_match = re.search(r"점수\s*[:：]?[\s]*(\d+)", text)
        change_match = re.search(r"변동\s*[:：]?[\s]*(상승|하락|유지)", text)
        reason_match = re.search(r"이유\s*[:：]?[\s]*(.*)", text)
        return {
            "score": int(score_match.group(1)) if score_match else None,
            "change": change_match.group(1) if change_match else "유지",
            "reason": reason_match.group(1).strip() if reason_match else ""
        }