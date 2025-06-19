from typing import Dict
from langgraph.graph import StateGraph, END
from core.chain import Chain
from core.llm import llm

AXIS = ["ei_score", "sn_score", "tf_score", "jp_score"]


def choose_axis_and_update(state: Dict) -> Dict:
    parsed_outputs = state["parsed_outputs"]
    current_scores = state["current_scores"]

    # 디버깅
    # print("LLM Raw Outputs")
    # for axis in AXIS:
    #     print(f"{axis}: {state['raw_outputs'].get(axis)}")

    print("Parsed Outputs")
    for axis in AXIS:
        print(f"{axis}: {state['parsed_outputs'].get(axis)}")

    # 점수가 None인 경우 현재 점수 유지
    for axis in AXIS:
        if parsed_outputs[axis].get("score") is None:
            parsed_outputs[axis]["score"] = current_scores[axis]
            parsed_outputs[axis]["change"] = "유지"


    # 상승 또는 하락 축만 필터링
    candidates = {
        axis: p for axis, p in parsed_outputs.items()
        if p.get("change") in ["상승", "하락"] and p.get("score") is not None
    }

    # 모든 축이 유지일 경우: 그대로 반환
    if not candidates:
        final_reason = '모든 축이 유지 상태로 판단되어 점수를 변경하지 않았습니다'
        state["final_result"] = {
            "updated_scores": current_scores,
            "chosen_axis": None,
            "reason": final_reason,
            "parsed_outputs": parsed_outputs
        }
        print("\n모든 축이 유지 상태") # 디버깅
        return state

    # 변화폭 가장 큰 축 선택
    diffs = {
        axis: abs(p["score"] - current_scores[axis])
        for axis, p in candidates.items()
    }
    chosen = max(diffs, key=diffs.get)



    # 각 축의 reason 수집
    axis_reason_map = {
        axis.replace("_score", "").upper(): parsed_outputs[axis]["reason"]
        for axis in AXIS
    }


    chosen_axis_label = chosen.replace("_score", "").upper()
    reasoning_prompt = (
        '''
        다음은 MBTI 4개 축에 대한 분석 결과입니다:

        {chr(10).join([f"**[{axis.replace('_score', '').upper()} 축]** {parsed_outputs[axis]['reason']}" for axis in AXIS])}

        위 분석 중 '{chosen_axis_label}' 축에서 점수 변화가 가장 컸습니다.
        이 축의 점수 변화 이유를 다음 형식에 맞춰 정확히 1문장으로 설명해주세요:

        **필수 형식:** ~한 이유로 '구체적인 문장'이 {chosen_axis_label}적으로 해석되어 점수를 ±N함.

        **출력 예시:**
        - 상징적 표현을 사용했다는 이유로 '머릿속이 우주처럼 느껴졌다'는 문장이 N적으로 해석되어 점수를 +5함.
        - 계획적 행동을 언급했다는 이유로 '일정을 미리 짜두었다'는 문장이 J적으로 해석되어 점수를 +5함.
        '''
        )

    # final_prompt = reasoning_prompt.format(
    #     parsed_output=parsed_output_text,
    #     chosen_axis_label=chosen_axis_label
    # )

    
    try:
        # 프롬프트 안나오고 결과만 디코딩
        input_ids = llm.tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(llm.model.device)
        input_length = input_ids.shape[1]

        output_ids = llm.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=llm.tokenizer.pad_token_id,
            eos_token_id=llm.tokenizer.eos_token_id,
        )

        # 프롬프트 이후 출력만 디코딩
        if len(output_ids[0]) > input_length:
            final_reason = llm.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
        else:
            final_reason = llm.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if chosen_axis_label not in final_reason:
            raise ValueError("LLM 요약 문장에 선택된 축이 명시되지 않았습니다.")

    except Exception as e:
        final_reason = f"[{chosen_axis_label} 축] 이유를 생성하는 도중 오류가 발생했습니다: {str(e)}"


    updated_scores = {
        axis: parsed_outputs[axis]["score"]
        for axis in AXIS
    }

    # 결과 저장
    state["final_result"] = {
        "updated_scores": updated_scores,
        "chosen_axis": chosen,
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