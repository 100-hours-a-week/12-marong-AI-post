from langchain.prompts import PromptTemplate

def get_axis_prompt_templates():
    axis_examples = {
        "ei_score": "여러 사람과 자연스럽게 대화를 나누는 모습은 외향(E) 성향으로 해석되어 점수를 상승함.",
        "sn_score": "구체적인 행동 묘사와 현재 상황을 강조한 문장은 감각(S) 성향으로 해석되어 점수를 하락함.",
        "tf_score": "상대방의 감정을 배려하며 선물을 선택한 행동은 감정(F) 성향으로 해석되어 점수를 상승함.",
        "jp_score": "미리 약속을 정하고 그것에 맞춰 행동한 모습은 판단(J) 성향으로 해석되어 점수를 하락함.",
    }

    axis_prompt_templates = {}
    for axis, (left, right, left_desc, right_desc) in {
        "ei_score": ("내향성(I)", "외향성(E)", "내면활동에 집중하고 조용함", "외부 자극을 통해 에너지를 얻고 사교적"),
        "sn_score": ("감각(S)", "직관(N)", "오감과 사실 중심", "비유와 상상 중심"),
        "tf_score": ("사고(T)", "감정(F)", "논리 중심 판단", "감정과 관계 중심 판단"),
        "jp_score": ("판단(J)", "인식(P)", "계획적이며 체계적", "유연하고 즉흥적"),
    }.items():
        axis_prompt_templates[axis] = PromptTemplate(
            input_variables=["user_feed", "current_score", "examples", "missions"],
            template=f"""
[기존점수]
{{current_score}}

[미션 내용]
유저는 다음과 같은 미션을 수행했습니다: {{missions}}

[유사 피드 예시]
{{examples}}

[유저 피드]
유저는 위 미션을 수행하며 다음과 같은 피드를 작성하였습니다: {{user_feed}}

[분석 방식]
- 유저 피드와 행동을 기반으로 '{left} vs {right}' 중 어떤 성향이 더 두드러지는지를 판단하세요.
- **미션 성공 여부**는 고려하지 마세요.
- 각 축의 성향 설명은 다음과 같습니다:
    - {left}: {left_desc}
    - {right}: {right_desc}
- 이유에 따라서 논리에 맞게 변화를 판단해주세요
- {right} 로 판단되면 **상승**, {left} 로 판단되면 **하락** 입니다.

- 아래와 같은 형식으로 결과를 반드시 작성하세요:
    점수: (숫자)
    변화: (하락/유지/상승)
    이유: 유저의 행동 또는 문장 중 특정 표현이 어떤 성향을 드러냈는지 분석한 **단 하나의 문장**을 작성하세요.

[예시]
{axis_examples[axis]}
"""
        )
    return axis_prompt_templates