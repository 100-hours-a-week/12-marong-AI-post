from typing import Dict
from langgraph.graph import StateGraph, END
from .chain import Chain
from .llm import llm
from .mbti_utils import AXIS
from .graph_result import FinalScoreDecider, NodeProcessor

class MBTIGraphRunner:
    def __init__(self, change_weight: int):
        self.change_weight = change_weight
        self.chain = Chain()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(state_schema=dict)

        axis_processor = NodeProcessor(self.chain, self.change_weight)

        for axis in AXIS:
            graph.add_node(axis, lambda state, axis=axis: axis_processor.run(axis, state))

        # 마지막 점수 보정 노드 추가
        final_decider = FinalScoreDecider(self.change_weight)
        graph.add_node("final_update", lambda state: final_decider.decide(state))

        # 노드 연결
        graph.set_entry_point("ei_score")
        graph.add_edge("ei_score", "sn_score")
        graph.add_edge("sn_score", "tf_score")
        graph.add_edge("tf_score", "jp_score")
        graph.add_edge("jp_score", "final_update")
        graph.add_edge("final_update", END)

        return graph.compile()

    def run(
        self,
        user_feed: str,
        current_scores: Dict[str, int],
        examples_text: str,
        missions_text: str
    ) -> Dict:
        initial_state = {
            "user_feed": user_feed,
            "current_scores": current_scores,
            "examples_text": examples_text,
            "raw_outputs": {},
            "parsed_outputs": {},
            "change_weight": self.change_weight,
            "missions_text": missions_text
        }

        final_state = self.graph.invoke(initial_state)
        final_result = final_state["final_result"]

        return {
            "axis_reason_map": {
                axis.replace("_score", "").upper(): final_result["parsed_outputs"][axis]["reason"]
                for axis in AXIS
            },
            "parsed_outputs": final_result["parsed_outputs"],
            "chosen_axis": final_result["chosen_axis"],
            "reason": final_result["reason"],
            "mbti": final_result["updated_scores"],
        }