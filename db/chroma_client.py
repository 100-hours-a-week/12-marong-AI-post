## main.py 실행시 연결되는 chromadb

import os
from dotenv import load_dotenv, find_dotenv
from chromadb.utils import embedding_functions
from chromadb import HttpClient

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# 기존 임베딩 함수
_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

def get_chroma_client():
    return HttpClient(
        host = os.getenv("CHROMA_HOST"),
        port = os.getenv("CHROMA_PORT"),
    )
    

def get_embedding_function():
    """문장 임베딩 함수 인스턴스 반환"""
    return _embedding_func