from chromadb import HttpClient
from chromadb.utils import embedding_functions

# 기존 임베딩 함수
_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

# HttpClient 인스턴스 생성
_client = HttpClient(
    host="localhost",
    port=8001,
    ssl=False
)

# 유사도 검색 collection name: mbti_feeds

def get_chroma_client():
    """Chroma DB HttpClient 인스턴스 반환"""
    return _client

def get_embedding_function():
    """문장 임베딩 함수 인스턴스 반환"""
    return _embedding_func



def get_user_latest_collection():
    """유저별 최신 MBTI 저장용 컬렉션"""
    return _client.get_or_create_collection(
        name="user_latest",
        embedding_function=None  # 메타데이터 전용
    )


def get_user_history_collection():
    """유저 MBTI 히스토리 저장용 컬렉션"""
    return _client.get_or_create_collection(
        name="user_history",
        embedding_function=None
    )
