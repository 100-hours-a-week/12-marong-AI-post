from typing import Dict
from langgraph.graph import StateGraph, END
from core.chain import Chain
from core.llm import llm
from core.mbti_utils import AXIS, AXIS_LABEL_MAP, AXIS_REVERSE_MAP, format_axis_reason, get_change_type
import re

AXIS = ["ei_score", "sn_score", "tf_score", "jp_score"]


def choose_axis_and_update(state: Dict) -> Dict:
    parsed_outputs = state["parsed_outputs"]
    current_scores = state["current_scores"]
    change_weight = state["change_weight"]

    print("Parsed Outputs")
    for axis in AXIS:
        print(f"{axis}: {state['parsed_outputs'].get(axis)}")

    # 점수가 None인 경우 현재 점수 유지
    for axis in AXIS:
        if parsed_outputs[axis].get("score") is None:
            parsed_outputs[axis]["score"] = current_scores[axis]
            parsed_outputs[axis]["change"] = "유지"

    # 가장 긴 텍스트를 가진 축 찾기
    longest_axis = max(AXIS, key = lambda a: len(parsed_outputs[a]["reason"] or ""))
    reason_text = parsed_outputs[longest_axis]["reason"]

    if "유지" in reason_text:
        parsed_outputs[longest_axis]['score'] = current_scores[longest_axis]
        parsed_outputs[longest_axis]['change'] = "유지"
        
    else:
        llm_prompt = f"""
    사용자의 피드에 기반한 다음 이유는 [{longest_axis.upper()} 축]에 대한 설명입니다:

    "{reason_text}"

    이 이유를 읽고, 이 축의 성향 점수를 어떻게 조정해야 할지 판단해주세요.
    다음 중 하나를 정확히 골라주세요: 상승 / 하락

    출력 형식 (반드시 지켜주세요):
    결정: 상승
    """.strip()
    
    try:
        input_ids = llm.tokenizer(llm_prompt, return_tensors="pt").input_ids.to(llm.model.device)
        input_length = input_ids.shape[1]

        output_ids = llm.model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=llm.tokenizer.pad_token_id,
            eos_token_id=llm.tokenizer.eos_token_id,
        )
        output_text = llm.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
        print("LLM 판단 결과", output_text)

        # 방향 결정
        match = re.search(f"결정\s*:\s*(상승|하락)", output_text)
        if not match:
            raise ValueError("LLM 응답에서 '상승' 또는 '하락'을 찾을 수 없습니다")
        direction = match.group(1)

    except Exception as e:
        print("LLM 판단 실패:", str(e))
        direction = "유지"

    # 점수 조정
    if direction == "상승":
        parsed_outputs[longest_axis]["score"] = min(100, current_scores[longest_axis] + change_weight)
        parsed_outputs[longest_axis]["change"] = "상승"
    else:  # 하락
        parsed_outputs[longest_axis]["score"] = max(0, current_scores[longest_axis] - change_weight)
        parsed_outputs[longest_axis]["change"] = "하락"

    # 나머지 축은 변경 없이 유지
    for axis in AXIS:
        if axis != longest_axis:
            parsed_outputs[axis]["score"] = current_scores[axis]
            parsed_outputs[axis]["change"] = "유지"

    updated_scores = {
        axis: parsed_outputs[axis]["score"] for axis in AXIS
    }

    # 결과 저장
    state["final_result"] = {
        "updated_scores": updated_scores,
        "chosen_axis": longest_axis,
        "reason": reason_text,
        "parsed_outputs": state["parsed_outputs"]
    }

    print("\n Final Result")
    print(state["final_result"])
    return state


def build_mbti_graph(chain: Chain, change_weight):
    graph = StateGraph(state_schema=dict)

    # 각 축별 노드 생성
    for axis in AXIS:
        def make_axis_func(axis, change_weight):  # 클로저 방지
            def node(state: Dict) -> Dict:
                result = chain.run(
                    axis=axis,
                    user_feed=state["user_feed"],
                    current_score=state["current_scores"][axis],
                    examples=state["examples_text"],
                    missions = state["missions_text"]
                )
                state["raw_outputs"][axis] = result["raw"]
                
                # 변화 계산
                parsed = result["parsed"]
                current = state["current_scores"][axis]
                predicted = parsed.get("score")

                if predicted is not None:
                    if predicted > current:
                        parsed["score"] = min(100, current + change_weight)
                        parsed["change"] = "상승"
                    elif predicted < current:
                        parsed["score"] = max(0, current - change_weight)
                        parsed["change"] = "하락"
                    else:
                        parsed["score"] = current
                        parsed["change"] = "유지"
                else:
                    parsed["score"] = current
                    parsed["change"] = "유지"

                state["parsed_outputs"][axis] = parsed
                return state
            return node
        graph.add_node(axis, make_axis_func(axis, change_weight))

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