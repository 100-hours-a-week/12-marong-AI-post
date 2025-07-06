import re
from typing import Dict
from core.llm import llm
from core.mbti_utils import AXIS


class FinalScoreDecider:
    def __init__(self, change_weight: int):
        self.change_weight = change_weight

    def decide(self, state: Dict) -> Dict:
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
            parsed_outputs[longest_axis]["score"] = min(100, current_scores[longest_axis] + self.change_weight)
            parsed_outputs[longest_axis]["change"] = "상승"
        else:  # 하락
            parsed_outputs[longest_axis]["score"] = max(0, current_scores[longest_axis] - self.change_weight)
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


class NodeProcessor:
    def __init__(self, chain, change_weight: int):
        self.chain = chain
        self.change_weight = change_weight

    def run(self, axis: str, state: Dict) -> Dict:
            result = self.chain.run(
                axis=axis,
                user_feed=state["user_feed"],
                current_score=state["current_scores"][axis],
                examples=state["examples_text"],
                missions=state["missions_text"]
            )
            state["raw_outputs"][axis] = result["raw"]

            parsed = result["parsed"]
            current = state["current_scores"][axis]
            predicted = parsed.get("score")

            if predicted is not None:
                if predicted > current:
                    parsed["score"] = min(100, current + self.change_weight)
                    parsed["change"] = "상승"
                elif predicted < current:
                    parsed["score"] = max(0, current - self.change_weight)
                    parsed["change"] = "하락"
                else:
                    parsed["score"] = current
                    parsed["change"] = "유지"
            else:
                parsed["score"] = current
                parsed["change"] = "유지"

            state["parsed_outputs"][axis] = parsed
            return state