import re, os
import torch
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from chroma_client_db import get_chroma_client, get_embedding_function  

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

#  CLOVAX 모델 로딩
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir,  "models", "hyperclovax-1.5b-instruct")
print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_path, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto"
)

# LangChain용 커스텀 래퍼
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

chroma_client = get_chroma_client()  
embedding_func = get_embedding_function()  
collection = chroma_client.get_or_create_collection(  
    name="mbti_feeds",  
    embedding_function=embedding_func  
)  

# 유사 피드 + MBTI 라벨 함께 가져오기
def retrieve_similar_texts(user_feed, top_k=5):
    results = collection.query(query_texts=[user_feed], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta.get("a_mbti", "UNKNOWN")) for doc, meta in zip(docs, metas)]

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

# 프롬프트 구성
axis_prompt_templates = {
    axis: PromptTemplate(
        input_variables=["user_feed", "current_score", "examples"],
        template=f"""
        너는 심리학자야.
        다음 피드 내용을 바탕으로 '{left} vs {right}' 중 어떤 성향이 더 강한지 판단해줘.

        기준 질문:
        - '{left}': {left_desc}
        - '{right}': {right_desc}

        기존 점수는 {{current_score}}야.

        성향 판단 방법:
        - '{right}' 성향에 더 가깝다고 판단되면 점수를 **상승**시켜. (점수 증가)
        - '{left}' 성향에 더 가깝다고 판단되면 점수를 **하락**시켜. (점수 하락)
        - 명확히 판단하기 어렵거나 중립적이라면 점수를 유지해.

        [유사 피드 예시]
        {{examples}}

        [유저 피드]
        {{user_feed}}

        결과 형식:
        점수: (숫자)
        변동: (상승/하락/유지)
        이유: (간단하고 논리적인 설명)
        """
    )
    for axis, (left, right, left_desc, right_desc) in {
        "ei_score": (
            "내향성(I)", "외향성(E)",
            "내면활동에 집중하고 조용하며 에너지 회복을 위해 혼자만의 시간이 필요",
            "외부 자극을 통해 에너지를 얻고 사교적이며 활동적"
        ),
        "sn_score": (
            "감각(S)", "직관(N)",
            "오감에 의존해 현재 사실과 디테일을 중시",
            "가능성과 비유를 통해 큰 그림을 보고 상상력을 발휘"
        ),
        "tf_score": (
            "사고(T)", "감정(F)",
            "객관적 사실과 논리에 기반해 판단",
            "정서와 인간관계를 중시해 판단"
        ),
        "jp_score": (
            "판단(J)", "인식(P)",
            "계획적이고 체계적으로 활동하며 빠른 결정을 선호",
            "유연하게 상황에 맞춰 즉흥적으로 행동하여 변화와 호기심을 추구"
        ),
    }.items()
} # 축 기준 ENFP

axis_chains = {
    axis: (
        axis_prompt_templates[axis]
        | llm
        | MBTIOutputParser()
    )
    for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]
}

def update_mbti(user_feed, current_scores, change_weight=5):
    examples = retrieve_similar_texts(user_feed)
    examples_text = "\n".join([f"- ({mbti}) {text}" for text, mbti in examples])

    # test 용
    raw_results={} # llm 출력 결과
    temp_results={} # 파상된 결과

    for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
        prompt_str = {
            "user_feed": user_feed,
            "current_score": current_scores[axis],
            "examples": examples_text
        }

        prompt_vars = {
            "user_feed":   user_feed,
            "current_score": current_scores[axis],
            "examples":    examples_text
        }

        prompt_str = axis_prompt_templates[axis].format(**prompt_vars)
        raw = llm.invoke(prompt_str)
        raw_results[axis] = raw

        parsed = axis_chains[axis].invoke(prompt_vars)
        temp_results[axis] = parsed

    # 변동 축 선택
    change_candidates = {axis: parsed for axis, parsed in temp_results.items() if parsed["change"] in ["상승", "하락"] and parsed.get("score") is not None}

    updated_result = {"mbti": {}, "changes": {}}
    if change_candidates:
        diffs = {
            axis: abs(parsed["score"] - current_scores[axis])
            for axis, parsed in change_candidates.items()
        }

        chosen = max(diffs, key=diffs.get)

        for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
            curr = current_scores[axis]
            if axis == chosen:
                diff = temp_results[axis]["score"] - curr
                delta = change_weight if diff > 0 else -change_weight
                new_score = min(100, max(0, curr + delta))
                change_flag = "상승" if diff > 0 else "하락"
                reason = temp_results[axis]["reason"]
            else:
                new_score = curr
                change_flag = "유지"
                reason = "변동없음"
                
            updated_result["mbti"][axis] = new_score
            updated_result ["changes"][axis] = {"변동": change_flag, "이유": reason}

    else: 
        for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
            updated_result["mbti"][axis] = current_scores[axis]
            updated_result["changes"][axis] = {"변동": "유지", "이유": "변동없음"}

    # 유사 예시 포함
    updated_result["similar_examples"] = [
        {"text": doc, "a_mbti": mbti_label}
        for doc, mbti_label in examples
    ]

    print("\n llm output")
    for axis, raw in raw_results.items():
        print(f"{axis}축 -> {raw}\n{'-'*40}")

    return updated_result