import re
from typing import Dict, List, Tuple
from db.chroma_client import get_chroma_client
from .llm import llm, MBTIOutputParser
from .prompts import get_axis_prompt_templates


# ChromaDB에서 유사 피드와 라벨 조회
class Retriever:
    def __init__(self, collection_name: str = 'mbti_feeds'):
        self.client = get_chroma_client()
        self.collection = self.client.get_collection(
            name = collection_name,
            #embedding_function = embedding_func
        )

    def get_similar(self, text: str, top_k: int = 5) -> List[Tuple[str, str]]:
        results = self.collection.query(query_texts=[text], n_results=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [(doc, meta.get("a_mbti", "UNKNOWN")) for doc, meta in zip(docs, metas)]


# 축별 chain, llm 호출, 파생
class Chain:
    def __init__(self):
        self.templates = get_axis_prompt_templates()
        self.chains = {
            axis: (self.templates[axis] | llm | MBTIOutputParser())
            for axis in self.templates
        }
        
    
    def run(self, axis: str, user_feed: str, current_score: int, examples: str) -> Dict:
        prompt = self.templates[axis].format(
            user_feed = user_feed,
            current_score = current_score,
            examples = examples
        )
        raw = llm.invoke(prompt)

        parsed = self.chains[axis].invoke({
            "user_feed": user_feed,
            "current_score": current_score,
            "examples": examples})
        
        # print(f"DEBUG[{axis}] LLM Raw Output:\n{raw}\n{'-'*40}")
        # print(f"DEBUG[{axis}] Parsed Output:\n{parsed}\n{'='*40}")

        return {"raw": raw, "parsed": parsed}


# mbti update
class MBTIUpdater:
    def __init__(self, change_weight: int = 5):
        self.retriever = Retriever()
        self.chain = Chain()
        self.change_weight = change_weight

    
    def update_mbti(self, user_feed: str, current_scores: Dict[str, int]) -> Dict:
        examples = self.retriever.get_similar(user_feed)
        examples_text = "\n".join([f"- ({mbti}) {doc}" for doc, mbti in examples])

        raw_outputs={} # llm 출력 결과
        parsed_outputs={} # 파상된 결과


        # chain 실행
        for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
            result = self.chain.run(axis, user_feed, current_scores[axis], examples_text)
            raw_outputs[axis] = result["raw"]
            parsed_outputs[axis] = result["parsed"]

        # 변화 축 선택 및 점수 조정
        candidates = {axis: p for axis, p in parsed_outputs.items()
                      if p.get("change") in ["상승", "하락"] and p.get("score") is not None}
        update_mbti = {"mbti": {}, "changes": {}, "similar_examples": []}

        if candidates:
            diffs = {axis: abs(p["score"] - current_scores[axis]) for axis, p in candidates.items()}
            chosen = max(diffs, key=diffs.get)

            for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
                curr = current_scores[axis]
                if axis == chosen:
                    diff = parsed_outputs[axis]["score"] - curr
                    delta = self.change_weight if diff > 0 else -self.change_weight
                    new_score = min(100, max(0, curr + delta))
                    change_flag = "상승" if diff > 0 else "하락"
                    reason = parsed_outputs[axis]["reason"]
                else:
                    new_score = curr
                    change_flag = "유지"
                    reason = "변화없음"

                update_mbti["mbti"][axis] = new_score
                update_mbti["changes"][axis] = {"변화": change_flag, "이유": reason}
        else:
            for axis in ["ei_score", "sn_score", "tf_score", "jp_score"]:
                update_mbti["mbti"][axis] = current_scores[axis]
                update_mbti["changes"][axis] = {"변화": "유지", "이유": "변화없음"}

        # 유사 예시 포함
        update_mbti["similar_examples"] = [{"text": doc, "a_mbti": mbti} for doc, mbti in examples]

        return update_mbti