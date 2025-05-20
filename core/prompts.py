from langchain.prompts import PromptTemplate

def get_axis_prompt_templates():
    axis_prompt_templates = {
        axis: PromptTemplate(
            input_variables=["user_feed", "current_score", "examples"],
            template=f"""
            너는 심리학자야.
            다음 피드 내용을 바탕으로 '{left} vs {right}' 중 어떤 성향이 더 강한지 판단해줘.

            기존 점수는 {{current_score}}야.

            성향 판단 방법:
            - '{right}' 성향에 더 가깝다고 판단되면 점수를 **상승**시켜. (점수 증가)
            - '{left}' 성향에 더 가깝다고 판단되면 점수를 **하락**시켜. (점수 하락)
            - 명확히 판단하기 어렵거나 중립적이라면 점수를 유지해.

            [유사 피드 예시]
            {{examples}}

            [유저 피드]
            {{user_feed}}

            결과 형식:
            점수: (숫자)
            변동: (상승/하락/유지)
            이유: (간단하고 논리적인 설명)
            """
        )
        for axis, (left, right, left_desc, right_desc) in {
            "ei_score": (
                "내향성(I)", "외향성(E)",
                "내면활동에 집중하고 조용하며 에너지 회복을 위해 혼자만의 시간이 필요",
                "외부 자극을 통해 에너지를 얻고 사교적이며 활동적"
            ),
            "sn_score": (
                "감각(S)", "직관(N)",
                "오감에 의존해 현재 사실과 디테일을 중시",
                "가능성과 비유를 통해 큰 그림을 보고 상상력을 발휘"
            ),
            "tf_score": (
                "사고(T)", "감정(F)",
                "객관적 사실과 논리에 기반해 판단",
                "정서와 인간관계를 중시해 판단"
            ),
            "jp_score": (
                "판단(J)", "인식(P)",
                "계획적이고 체계적으로 활동하며 빠른 결정을 선호",
                "유연하게 상황에 맞춰 즉흥적으로 행동하여 변화와 호기심을 추구"
            ),
        }.items()
    } # 축 기준 ENFP

    return axis_prompt_templates