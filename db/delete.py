## ChromaDB reset

from db.chroma_client import get_chroma_client

# ChromaDB 클라이언트 불러오기
client = get_chroma_client()

# 삭제할 컬렉션 이름 리스트
to_delete = [
    "user_latest",
    "user_history"
]

# 컬렉션 삭제
for name in to_delete:
    try:
        client.delete_collection(name=name)
        print(f" 컬렉션 '{name}' 삭제 완료")
    except Exception as e:
        print(f" 컬렉션 '{name}' 삭제 실패:", e)
