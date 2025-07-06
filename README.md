# 12-marong-AI-post
<img width="1273" alt="스크린샷 2025-06-01 오전 5 04 20" src="https://github.com/user-attachments/assets/1bfffd0c-a2bd-4d18-bdc7-f8f3106e470a" />


# 📍 Overview

- 프로젝트 이름: Marong
- 프로젝트 설명: 마니또 기반 SNS 서비스


## 📍 서비스 아키텍처 다이어그램
 <img width="600" alt="스크린샷 2025-04-29 오후 8 25 00" src="https://github.com/user-attachments/assets/ad186ffa-ea3a-4889-b7f0-aa1d1650e3cd" />

## 📍 피드기반 행동분석
- 사용자가 업로드한 피드 텍스트를 RAG 기반으로 분석해, 현 MBTI 점수를 보정 및 업데이트 진행
> - RAG 모듈 (문서로딩 -> Splitting -> Embedding -> Retriever -> Top-k
> - PromptTemplate -> LLM Chain -> Score Parser -> mbti 업데이트


## 📍 API 명세
> [POST] `/mbti/update`
#### Request Body:
```
{
    "id": "user123",
    "eiScore": 50,
    "snScore": 50,
    "tfScore": 50,
    "jpScore": 50, 
    "hobbies": ["게임", "운동"],
    "content": "마니또에게 인사하는 미션 완료!"
}
```

#### Response Body:
```
{
  "user_id" : "user123",
  "message" : "update_success",
  "data" : {
    "mbti_scores": {"e": 80, "s": 60, "t": 45, "j": 70}, 
    "hobby": "운동",
    "user_feed": "마니또에게 인사하는 미션 완료!"
   }
}
```


## 📍 모델 선정 및 사용 이유
- 사용 모델: `hyperclovax-1.5b-instruct`
- 이유: 타 모델과 비교해보았을 때, 적은 파라미터로 높은 한국어 이해도와 빠른 추론 성능

## 📍 추후 개선 사항들
- 입력받는 사용자 취미도 함께 반영 후 mbti 업데이트
- 성능 고도화
- QLoRA를 활용한 llm 파인튜닝 (번역된 kaggle MBTI500 사용)
