import re
from typing import Dict, List, Tuple
from db.chroma_client import get_chroma_client
from .graph import MBTIGraphRunner
from .chain import Chain

# ChromaDB에서 유사 피드와 라벨 조회
class Retriever:
    def __init__(self, collection_name: str = 'mbti_feeds'):
        self.client = get_chroma_client()
        self.collection = self.client.get_collection(name = collection_name,)

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

    
    def update_mbti(self, user_feed: str, current_scores: Dict[str, int], missions_text: str, original_scores: Dict[str, int]) -> Dict:
        examples = self.retriever.get_similar(user_feed)
        examples_text = "\n".join([f"- ({mbti}) {doc}" for doc, mbti in examples])

        runner = MBTIGraphRunner(change_weight=self.change_weight)

        # langGraph로 최종 점수 결정
        final_result =runner.run(
            user_feed = user_feed,
            current_scores=current_scores,
            examples_text = examples_text,
            missions_text = missions_text
        )

        changed_axis = final_result["chosen_axis"] 
        previous_score = original_scores[f"{changed_axis}"]  
        current_score = final_result["mbti"][f"{changed_axis}"]  

        return {
            "mbti": final_result["mbti"],
            "changed_axis": changed_axis,  
            # "previous_score": previous_score,  
            "current_score": current_score,  
            "final_reason": final_result.get("reason", "이유 없음"),
            "original_score": original_scores
        }