# chroma server 실행 파일

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

def run_chroma():
    cmd = os.getenv("CHROMA_CMD", "chroma")
    host = os.getenv("CHROMA_HOST","0.0.0.0")
    port = os.getenv("CHROMA_PORT", "8001")
    path = os.getenv("CHROMA_PATH", "")

    
    try:
        subprocess.run([
            cmd,
            "run",
            "--host", host,
            "--port", port,
            "--path", path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("Chroma 실행 중 오류 발생:", e)
    except FileNotFoundError:
        print("'chroma' 명령어를 찾을 수 없습니다. PATH에 등록되어 있는지 확인하세요.")

if __name__ == "__main__":
    run_chroma()