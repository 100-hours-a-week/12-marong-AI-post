from typing import Dict
from langgraph.graph import StateGraph, END
from core.chain import Chain
from core.llm import llm
from core.mbti_utils import AXIS
from .graph_result import choose_axis_and_update, make_axis_node


def build_mbti_graph(chain: Chain, change_weight: int):
    graph = StateGraph(state_schema=dict)

    # 각 축별 노드 생성
    for axis in AXIS:
        graph.add_node(axis, make_axis_node(axis, chain, change_weight))
        

    # 마지막 점수 보정 노드 추가
    graph.add_node("final_update", choose_axis_and_update)

    # 노드 연결
    graph.set_entry_point("ei_score")
    graph.add_edge("ei_score", "sn_score")
    graph.add_edge("sn_score", "tf_score")
    graph.add_edge("tf_score", "jp_score")
    graph.add_edge("jp_score", "final_update")
    graph.add_edge("final_update", END)

    return graph.compile()

def run_mbti_update_with_graph(
    user_feed: str,
    current_scores: Dict[str, int],
    examples_text: str,
    change_weight: int,
    missions_text: str
):
    chain = Chain()
    app = build_mbti_graph(chain, change_weight)

    initial_state = {
        "user_feed": user_feed,
        "current_scores": current_scores,
        "examples_text": examples_text,
        "raw_outputs": {},
        "parsed_outputs": {},
        "change_weight": change_weight,
        "missions_text": missions_text
    }

    final_state = app.invoke(initial_state)
    final_result = final_state["final_result"]

    return {
        "axis_reason_map": {
            axis.replace("_score", "").upper(): final_state["parsed_outputs"][axis]["reason"]
            for axis in AXIS
        },
        "parsed_outputs": final_result["parsed_outputs"],
        "chosen_axis": final_result["chosen_axis"],
        "reason": final_result["reason"], 
        "mbti": final_result["updated_scores"],  # 최종 점수도 포함
    }