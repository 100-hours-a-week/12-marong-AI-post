## chroma db에 업데이트 잘 됐는지 확인하는 파일

from chroma_client_db import (
    get_mbti_latest_collection,
    get_mbti_history_collection,
    get_hobby_latest_collection,
    get_hobby_history_collection
)

# MBTI 컬렉션
latest_mbti = get_mbti_latest_collection()
history_mbti = get_mbti_history_collection()

# 취미 컬렉션
latest_hobby = get_hobby_latest_collection()
history_hobby = get_hobby_history_collection()

print("=== MBTI Latest ===")
print(latest_mbti.peek(limit=5))

print("\n=== MBTI History ===")
print(history_mbti.peek(limit=5))

print("\n=== Hobby Latest ===")
print(latest_hobby.peek(limit=5))

print("\n=== Hobby History ===")
print(history_hobby.peek(limit=5))

