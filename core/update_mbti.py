import re
from typing import Dict, List, Tuple
from db.chroma_client import get_chroma_client
from .graph import run_mbti_update_with_graph
from .chain import Chain

# 점수 업데이트
class MBTIScoreUpdater:
    axis_map = {
        "EI": "ei_score",
        "SN": "sn_score",
        "TF": "tf_score",
        "JP": "jp_score"
    }

    @staticmethod
    def apply_reason(reason: str, prev_scores: Dict[str, int]) -> Dict[str, int]:
        axis_match = re.search(r"\[([A-Z]{2}) 축\]", reason)
        if not axis_match:
            raise ValueError("축명을 찾을 수 없습니다.")
        axis_key = axis_match.group(1)
        score_key = MBTIScoreUpdater.axis_map.get(axis_key)

        score_change_match = re.search(r"점수를\s*([+-]?\d+)\s*함", reason)
        if not score_change_match:
            raise ValueError("점수 변화를 찾을 수 없습니다.")
        delta = int(score_change_match.group(1))

        new_score = max(0, min(100, prev_scores[score_key] + delta))
        updated_scores = prev_scores.copy()
        updated_scores[score_key] = new_score
        return updated_scores


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


# mbti update
class MBTIUpdater:
    def __init__(self, change_weight: int = 5):
        self.retriever = Retriever()
        self.chain = Chain()
        self.change_weight = change_weight

    
    def update_mbti(self, user_feed: str, current_scores: Dict[str, int]) -> Dict:
        examples = self.retriever.get_similar(user_feed)
        examples_text = "\n".join([f"- ({mbti}) {doc}" for doc, mbti in examples])

        # langGraph로 최종 점수 결정
        final_result = run_mbti_update_with_graph(
            user_feed = user_feed,
            current_scores=current_scores,
            examples_text = examples_text,
            change_weight = self.change_weight
        )

        axis_reason_map = final_result["axis_reason_map"]  
        parsed_outputs = final_result["parsed_outputs"]

        updated_scores = current_scores.copy()
        changes = {}

        for axis_key, reason in axis_reason_map.items():
            axis_code = axis_key.upper()
            score_key = {
                "EI": "ei_score",
                "SN": "sn_score",
                "TF": "tf_score",
                "JP": "jp_score"
            }[axis_code]

            # 점수 업데이트
            predicted_score = parsed_outputs[score_key]["score"]
            updated_scores[score_key] = predicted_score

            # 변화 방향
            change = parsed_outputs[score_key]["change"]

            if score_key == final_result["chosen_axis"]:
                변화 = "상승" if change == "상승" else "하락"
                changes[score_key] = {"변화": 변화, "이유": reason}
            else:
                changes[score_key] = {"변화": "유지", "이유": "변화없음"}

        return {
            "mbti": updated_scores,
            "reason": changes,
        }