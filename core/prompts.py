from langchain.prompts import PromptTemplate

def get_axis_prompt_templates():
    axis_prompt_templates = {
        axis: PromptTemplate(
            input_variables=["user_feed", "current_score", "examples"],
            template=f"""
            [기존점수]
            {{current_score}}

            [유사 피드]
            {{examples}}

            [유저 피드]
            {{user_feed}}



            [예측방식]
            1) 1. '{left}' : **-5점**, '유지' : +0점, '{right}' : **+5점**  
            2) 기존 점수({{current_score}}) 에 위 변화량을 더해서 최종 점수를 계산  

            예시1) 이전 점수 40 → 예측 45 (상승)  
            예시2) 이전 점수 70 → 예측 70 (유지)  
            예시3) 이전 점수 80 → 예측 75 (하락)    
                    

            결과 형식:
            점수: (숫자)
            변화: (하락/유지/상승)
            이유: ~한 이유로 ~문장이 ~하게 해석되어 점수를 ~함.' 형태로 요약해주세요. 문장은 하나만 사용하세요.
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