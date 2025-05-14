## chroma server 실행 파일

import subprocess
from chromadb import HttpClient
import os
from dotenv import load_dotenv

load_dotenv()

def run_chroma():
    cmd  = os.getenv("CHROMA_CMD")
    host = os.getenv("CHROMA_HOST")
    port = os.getenv("CHROMA_PORT")
    path = os.getenv("CHROMA_PATH")

    missing = [k for k, v in {
        "CHROMA_CMD": cmd,
        "CHROMA_HOST": host,
        "CHROMA_PORT": port,
        "CHROMA_PATH": path
    }.items() if not v]
    if missing:
        raise RuntimeError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing)}")


    try:
        subprocess.run([
            cmd,
            "run",
            "--host", str(host),
            "--port", str(port),
            "--path", str(path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("Chroma 실행 중 오류 발생:", e)
    except FileNotFoundError:
        print("'chroma' 명령어를 찾을 수 없습니다. PATH에 등록되어 있는지 확인하세요.")

    return HttpClient(host=host, port=port)

if __name__ == "__main__":
    run_chroma()