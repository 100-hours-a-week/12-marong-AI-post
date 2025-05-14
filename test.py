# 결과 비교용 gemini

import os
import uuid
import re
import torch
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from chroma_client_db import get_chroma_client, get_embedding_function

# 환경변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# LangChain Gemini LLM 인스턴스
llm = ChatGoogleGenerativeAI(  
    model="gemini-2.0-flash",  
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)

# ChromaDB 연결
chroma_client = get_chroma_client()
embedding_func = get_embedding_function()
collection = chroma_client.get_or_create_collection(
    name="mbti_feeds",
    embedding_function=embedding_func
)

# 유사 피드 + MBTI 라벨 함께 가져오기
def retrieve_similar_texts(user_feed: str, top_k: int = 5):
    results = collection.query(query_texts=[user_feed], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta.get("a_mbti", "UNKNOWN")) for doc, meta in zip(docs, metas)]

# 출력 파서
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

# 프롬프트 템플릿 정의
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
- '{right}' 성향에 더 가깝다고 판단되면 점수를 **상승**시켜.
- '{left}' 성향에 더 가깝다고 판단되면 점수를 **하락**시켜.
- 중립적이라면 점수를 유지해.

[유사 피드 예시]
{{examples}}

[유저 피드]
{{user_feed}}

결과 형식:
점수: (숫자)
변동: (상승/하락/유지)
이유: (간단한 설명)
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
            "유연하게 상황에 맞춰 즉흥적으로 행동하며 변화와 호기심을 추구"
        ),
    }.items()
}

# LangChain 체인 맵핑
axis_chains = {
    axis: (axis_prompt_templates[axis] | llm | MBTIOutputParser())
    for axis in axis_prompt_templates
}

# MBTI 업데이트 함수
def update_mbti(user_feed: str, current_scores: dict, change_weight: int = 5):
    # 1) 유사 예시 준비
    examples = retrieve_similar_texts(user_feed)
    examples_text = "\n".join([f"- ({mbti}) {text}" for text, mbti in examples])

    raw_results = {}
    parsed_results = {}

    # 2) 축별 호출 및 파싱
    for axis in axis_chains:
        prompt_vars = {
            "user_feed": user_feed,
            "current_score": current_scores[axis],
            "examples": examples_text
        }
        raw = llm.invoke(axis_prompt_templates[axis].format(**prompt_vars))
        raw_results[axis] = raw
        parsed = axis_chains[axis].invoke(prompt_vars)
        parsed_results[axis] = parsed

    # 3) 변화 후보 → 최대 변화 축 선택
    candidates = {
        axis: p for axis, p in parsed_results.items()
        if p["change"] in ["상승", "하락"] and p["score"] is not None
    }
    updated = {"mbti": {}, "changes": {}}

    if candidates:
        diffs = {axis: abs(p["score"] - current_scores[axis]) for axis, p in candidates.items()}
        chosen = max(diffs, key=diffs.get)

        for axis in axis_chains:
            curr = current_scores[axis]
            if axis == chosen:
                diff = parsed_results[axis]["score"] - curr
                delta = change_weight if diff > 0 else -change_weight
                new_score = max(0, min(100, curr + delta))
                flag = "상승" if diff > 0 else "하락"
                reason = parsed_results[axis]["reason"]
            else:
                new_score, flag, reason = curr, "유지", "변동없음"
            updated["mbti"][axis] = new_score
            updated["changes"][axis] = {"변동": flag, "이유": reason}
    else:
        for axis in axis_chains:
            updated["mbti"][axis] = current_scores[axis]
            updated["changes"][axis] = {"변동": "유지", "이유": "변동없음"}

    # 4) 유사 예시 포함
    updated["similar_examples"] = [{"text": d, "a_mbti": m} for d, m in examples]

    # 디버그 출력
    print("\nllm outputs:")
    for axis, raw in raw_results.items():
        print(f"{axis}: {raw}\n{'-'*30}")

    return updated

# (나머지) MBTIUpdateService, fetch_all_users, fetch_prev_data, run 등 기존 코드와 결합해서 사용
