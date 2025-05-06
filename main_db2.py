## main db 하드코딩된 파일, chroma db에 저장 제대로 되는건 확인 완료

import uuid
from datetime import datetime

from mbti_update_db import update_mbti
from chroma_client_db import (
    get_chroma_client,
    get_mbti_latest_collection,
    get_mbti_history_collection,
    get_hobby_latest_collection,
    get_hobby_history_collection
)

# 사용자 입력
user_id = input(" 사용자 ID 입력: ")

# 하드코딩된 입력값들 (DB 대체)
feed_text = "오늘 날씨가 좋아서 산책했어! 기분이 정말 좋아 "
prev_scores = {"E": 55, "S": 60, "T": 45, "J": 70}
prev_hobby = "등산"
DEFAULT_CHANGE_WEIGHT = 5


# ChromaDB 연결 및 컬렉션 준비
client = get_chroma_client()
mbti_latest_col = get_mbti_latest_collection()
mbti_history_col = get_mbti_history_collection()
hobby_latest_col = get_hobby_latest_collection()
hobby_history_col = get_hobby_history_collection()


# 기존 사용자 여부 판단
mbti_recs = mbti_latest_col.get(
    where={"user_id": user_id}, limit=1, include=["embeddings"]
)
hobby_recs = hobby_latest_col.get(
    where={"user_id": user_id}, limit=1, include=["metadatas"]
)



# MBTI 초기값 덮어쓰기 (기존 유저라면)
if mbti_recs.get("ids"):
    vec = mbti_recs["embeddings"][0]
    prev_scores = {"E": vec[0], "S": vec[1], "T": vec[2], "J": vec[3]}

# Hobby 초기값 덮어쓰기 (기존 유저라면)
if hobby_recs.get("ids"):
    prev_hobby = hobby_recs["metadatas"][0].get("hobby_name", prev_hobby)




# MBTI 점수 업데이트
updated = update_mbti(feed_text, prev_scores, DEFAULT_CHANGE_WEIGHT)
new_vec = [
    updated["mbti"]["E"],
    updated["mbti"]["S"],
    updated["mbti"]["T"],
    updated["mbti"]["J"],
]
timestamp = datetime.utcnow().isoformat()



# ChromaDB에 결과 저장

# MBTI 기록 저장
mbti_history_col.add(
    ids=[str(uuid.uuid4())],
    metadatas=[{
        "user_id": user_id,
        "timestamp": timestamp,
        "hobby": prev_hobby or ""
    }],
    documents=[""],
    embeddings=[new_vec]
)

mbti_latest_col.upsert(
    ids=[user_id],
    metadatas=[{
        "user_id": user_id,
        "updated_at": timestamp,
        "hobby": prev_hobby
    }],
    documents=[""],
    embeddings=[new_vec]
)

# Hobby 기록 저장
if prev_hobby:
    hobby_history_col.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{
            "user_id": user_id,
            "timestamp": timestamp,
            "hobby_name": prev_hobby
        }],
        documents=[""],
        embeddings=[[0.0]]
    )
    hobby_latest_col.upsert(
        ids=[user_id],
        metadatas=[{
            "user_id": user_id,
            "hobby_name": prev_hobby,
            "updated_at": timestamp
        }],
        documents=[""],
        embeddings=[[0.0]]
    )

# 확인용 출력
print(f"\n사용자 [{user_id}]의 MBTI가 업데이트되었습니다!")
print(f" MBTI 점수: {updated['mbti']}")
print(f"취미: {prev_hobby}")
