from typing import Dict
from langgraph.graph import StateGraph, END
from core.chain import Chain
from core.llm import llm
import re

AXIS = ["ei_score", "sn_score", "tf_score", "jp_score"]


def choose_axis_and_update(state: Dict) -> Dict:
    parsed_outputs = state["parsed_outputs"]
    current_scores = state["current_scores"]

    print("Parsed Outputs")
    for axis in AXIS:
        print(f"{axis}: {state['parsed_outputs'].get(axis)}")

    # 점수가 None인 경우 현재 점수 유지
    for axis in AXIS:
        if parsed_outputs[axis].get("score") is None:
            parsed_outputs[axis]["score"] = current_scores[axis]
            parsed_outputs[axis]["change"] = "유지"

    # 분석 요약 텍스트 생성
    reasoning_input = ""
    for axis in AXIS:
        axis_label = axis.replace("_score", "").upper()
        score_before = current_scores[axis]
        score_after = parsed_outputs[axis]["score"]
        change = parsed_outputs[axis]["change"]
        reason = parsed_outputs[axis]["reason"]

        reasoning_input += f"[{axis_label} 축]\n"
        reasoning_input += f"- 이전 점수: {score_before}, 예측 점수: {score_after} ({change})\n"
        reasoning_input += f"- 이유: {reason}\n\n"


    final_prompt = f"""
    다음은 사용자의 MBTI 각 축(EI, SN, TF, JP)에 대한 점수 변화 및 이유입니다:

    {reasoning_input}

    이 중 점수 변화가 있었고(reason과 change가 논리적으로 일치하는 경우),  
    가장 설득력 있는 **하나의 축**만 선택하세요.

    **출력 형식 (반드시 지켜주세요)**:
    선택된 축: SN
    """.strip()

    print("\n LLM Prompt:")
    print(final_prompt)
    
    try:
        # 프롬프트 안나오고 결과만 디코딩
        input_ids = llm.tokenizer(final_prompt, return_tensors="pt").input_ids.to(llm.model.device)
        input_length = input_ids.shape[1]

        output_ids = llm.model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=llm.tokenizer.pad_token_id,
            eos_token_id=llm.tokenizer.eos_token_id,
        )
        output_text = llm.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
        print("LLM output", output_text)

        chosen_match = re.search(r"선택된 축\s*:\s*(EI|SN|TF|JP)", output_text)

        if not chosen_match :
            raise ValueError("LLM 응답에서 축 또는 이유를 추출할 수 없습니다.")

        chosen_label = chosen_match.group(1)
        chosen_axis = f"{chosen_label.lower()}_score" 
        final_reason = parsed_outputs[chosen_axis]["reason"]

    except Exception as e:
        final_reason = f"이유 생성 중 오류 발생: {str(e)}"
        chosen_axis = None


    updated_scores = {
        axis: parsed_outputs[axis]["score"]
        for axis in AXIS
    }

    # 결과 저장
    state["final_result"] = {
        "updated_scores": updated_scores,
        "chosen_axis": chosen_axis,
        "reason": final_reason,
        "parsed_outputs": state["parsed_outputs"]
    }

    print("\n Final Result")
    print(state["final_result"])
    return state


def build_mbti_graph(chain: Chain):
    graph = StateGraph(state_schema=dict)

    # 각 축별 노드 생성
    for axis in AXIS:
        def make_axis_func(axis):  # 클로저 방지
            def node(state: Dict) -> Dict:
                result = chain.run(
                    axis=axis,
                    user_feed=state["user_feed"],
                    current_score=state["current_scores"][axis],
                    examples=state["examples_text"]
                )
                state["raw_outputs"][axis] = result["raw"]
                
                # 변화 계산
                parsed = result["parsed"]
                current = state["current_scores"][axis]
                predicted = parsed.get("score")

                if predicted is not None:
                    if predicted > current:
                        parsed["change"] = "상승"
                    elif predicted < current:
                        parsed["change"] = "하락"
                    else:
                        parsed["change"] = "유지"
                else:
                    parsed["score"] = current
                    parsed["change"] = "유지"

                state["parsed_outputs"][axis] = parsed
                return state
            return node
        graph.add_node(axis, make_axis_func(axis))

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
    change_weight: int
):
    chain = Chain()
    app = build_mbti_graph(chain)

    initial_state = {
        "user_feed": user_feed,
        "current_scores": current_scores,
        "examples_text": examples_text,
        "raw_outputs": {},
        "parsed_outputs": {},
        "change_weight": change_weight,
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
        "reason": final_result["reason"],  # 혹시 추후 reason도 활용할 수 있게 포함
        "mbti": final_result["updated_scores"],  # 최종 점수도 포함
    }