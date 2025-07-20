import re
from typing import Dict
from core.llm import llm
from core.mbti_utils import AXIS

class FinalScoreDecider:
    def __init__(self, current_scores: Dict[str, int], change_weight: int):
        self.current_scores = current_scores
        self.change_weight = change_weight

    async def decide(self, state: Dict) -> Dict:
        parsed_outputs = state.get("parsed_outputs", {})
        current_scores = self.current_scores
        change_weight = self.change_weight

        print("[DEBUG] FinalScoreDecider: parsed_outputs keys =", list(parsed_outputs.keys()))

        # 누락된 축은 유지로 설정
        for axis in AXIS:
            if axis not in parsed_outputs:
                parsed_outputs[axis] = {
                    "score": current_scores[axis],
                    "change": "유지",
                    "reason": ""
                }

        # 가장 긴 reason을 기준으로 대표 축 선정
        longest_axis = max(AXIS, key=lambda a: len(parsed_outputs[a]["reason"] or ""))
        reason_text = parsed_outputs[longest_axis]["reason"]

        # 대표 축의 변경 방향 판단
        if "유지" in reason_text:
            direction = "유지"
        else:
            try:
                llm_prompt = f"""
사용자의 피드에 기반한 다음 이유는 [{longest_axis.upper()} 축]에 대한 설명입니다:

"{reason_text}"

이 이유를 읽고, 이 축의 성향 점수를 어떻게 조정해야 할지 판단해주세요.
다음 중 하나를 정확히 골라주세요: 상승 / 하락

출력 형식 (반드시 지켜주세요):
결정: 상승
""".strip()

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

                match = re.search(r"결정\s*:\s*(상승|하락)", output_text)
                direction = match.group(1) if match else "유지"

            except Exception as e:
                print("LLM 판단 실패:", str(e))
                direction = "유지"

        # 점수 반영
        if direction == "상승":
            parsed_outputs[longest_axis]["score"] = min(100, current_scores[longest_axis] + change_weight)
            parsed_outputs[longest_axis]["change"] = "상승"
        elif direction == "하락":
            parsed_outputs[longest_axis]["score"] = max(0, current_scores[longest_axis] - change_weight)
            parsed_outputs[longest_axis]["change"] = "하락"
        else:
            parsed_outputs[longest_axis]["score"] = current_scores[longest_axis]
            parsed_outputs[longest_axis]["change"] = "유지"

        # 나머지 축은 무조건 유지
        for axis in AXIS:
            if axis != longest_axis:
                parsed_outputs[axis]["score"] = current_scores[axis]
                parsed_outputs[axis]["change"] = "유지"

        updated_scores = {axis: parsed_outputs[axis]["score"] for axis in AXIS}

        state["final_result"] = {
            "updated_scores": updated_scores,
            "chosen_axis": longest_axis,
            "reason": reason_text,
            "parsed_outputs": parsed_outputs
        }

        print("[DEBUG] FinalScoreDecider: final_result 생성 완료 ")
        return state


class NodeProcessor:
    def __init__(self, chain):
        self.chain = chain

    async def run(self, axis: str, state: Dict, user_feed: str, current_score: int, examples: str, missions: str, change_weight: int) -> Dict:
        result = await self.chain.arun(
            axis=axis,
            user_feed=user_feed,
            current_score=current_score,
            examples=examples,
            missions=missions
        )

        parsed = result["parsed"]
        predicted = parsed.get("score")

        if predicted is not None:
            if predicted > current_score:
                parsed["score"] = min(100, current_score + change_weight)
                parsed["change"] = "상승"
            elif predicted < current_score:
                parsed["score"] = max(0, current_score - change_weight)
                parsed["change"] = "하락"
            else:
                parsed["score"] = current_score
                parsed["change"] = "유지"
        else:
            parsed["score"] = current_score
            parsed["change"] = "유지"

        # parsed_outputs 딕셔너리 초기화 여부 확인 후 병합
        if "parsed_outputs" not in state:
            state["parsed_outputs"] = {}

        state["parsed_outputs"][axis] = parsed

        print(f"[DEBUG] NodeProcessor: axis '{axis}' 실행 완료, 결과 저장됨")
        return state