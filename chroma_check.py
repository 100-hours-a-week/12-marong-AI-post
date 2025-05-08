## chroma db에 업데이트 잘 됐는지 확인하는 파일

from chroma_client_db import (
    get_user_latest_collection,
    get_user_history_collection,
)

# MBTI 컬렉션
latest_mbti = get_user_latest_collection()
history_mbti = get_user_history_collection()


print("=== USER Latest ===")
print(latest_mbti.peek(limit=100))

print("\n=== USER History ===")
print(history_mbti.peek(limit=100))

