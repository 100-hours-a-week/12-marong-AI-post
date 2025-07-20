from typing import TypedDict, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from .chain import Chain
from .mbti_utils import AXIS
from .graph_result import FinalScoreDecider, NodeProcessor

def merge_dicts(old: Dict, new: Dict) -> Dict:
    return {**old, **new}

class GraphState(TypedDict):
    parsed_outputs: Annotated[Dict[str, Any], merge_dicts]
    final_result: Dict

class MBTIGraphRunner:
    def __init__(self, change_weight: int):
        self.change_weight = change_weight
        self.chain = Chain()

    def _build_graph(self, user_feed, current_scores, examples_text, missions_text):
        graph = StateGraph(GraphState)
        axis_processor = NodeProcessor(self.chain)
        final_decider = FinalScoreDecider(current_scores, self.change_weight)

        graph.add_node("start", lambda state: state)
        graph.set_entry_point("start")

        def make_axis_node(axis_name):
            async def axis_node(state):
                return await axis_processor.run(
                    axis=axis_name,
                    state=state,
                    user_feed=user_feed,
                    current_score=current_scores[axis_name],
                    examples=examples_text,
                    missions=missions_text,
                    change_weight=self.change_weight
                )
            return axis_node

        for axis in AXIS:
            graph.add_node(axis, make_axis_node(axis))
            graph.add_edge("start", axis)        
            graph.add_edge(axis, "final_update") 

        async def final_node(state):
            return await final_decider.decide(state)

        graph.add_node("final_update", final_node)
        graph.add_edge("final_update", END)

        return graph.compile()
    

    async def run(self, user_feed: str, current_scores: Dict[str, int], examples_text: str, missions_text: str) -> Dict:
        self.graph = self._build_graph(user_feed, current_scores, examples_text, missions_text)

        initial_state = {
            "parsed_outputs": {},
        }

        final_state = await self.graph.ainvoke(initial_state)
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