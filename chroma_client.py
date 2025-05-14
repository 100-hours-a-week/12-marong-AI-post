# main.py 실행시 연결되는 chromadb

import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb import HttpClient

# 기존 임베딩 함수
_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

load_dotenv()


def get_chroma_client():
    return HttpClient(
        host = os.getenv("CHROMA_HOST"),
        port = os.getenv("CHROMA_PORT"),
    )
    


def get_embedding_function():
    """문장 임베딩 함수 인스턴스 반환"""
    return _embedding_func



def get_user_latest_collection():
    return get_chroma_client().get_or_create_collection(
        name="user_latest", embedding_function=None
    )

def get_user_history_collection():
    return get_chroma_client().get_or_create_collection(
        name="user_history", embedding_function=None
    )