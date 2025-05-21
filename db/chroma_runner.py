# ## chroma server 실행 파일

# import subprocess
# from chromadb import HttpClient
# import os
# from dotenv import load_dotenv, find_dotenv

# dotenv_path = find_dotenv()
# load_dotenv(dotenv_path)

# def run_chroma():
#     cmd  = os.getenv("CHROMA_CMD")
#     host = os.getenv("CHROMA_HOST")
#     port = os.getenv("CHROMA_PORT")
#     path = os.getenv("CHROMA_PATH")

#     missing = [k for k, v in {
#         "CHROMA_CMD": cmd,
#         "CHROMA_HOST": host,
#         "CHROMA_PORT": port,
#         "CHROMA_PATH": path
#     }.items() if not v]
#     if missing:
#         raise RuntimeError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing)}")


#     try:
#         subprocess.run([
#             cmd,
#             "run",
#             "--host", str(host),
#             "--port", str(port),
#             "--path", str(path)
#         ], check=True)
#     except subprocess.CalledProcessError as e:
#         print("Chroma 실행 중 오류 발생:", e)
#     except FileNotFoundError:
#         print("'chroma' 명령어를 찾을 수 없습니다. PATH에 등록되어 있는지 확인하세요.")

#     return HttpClient(host=host, port=port)

# if __name__ == "__main__":
#     run_chroma()

from dotenv import load_dotenv
import subprocess, os

def run_chroma():
    try:
        load_dotenv()
        
        # 환경 변수 로드
        CHROMA_PORT = os.getenv("CHROMA_PORT")
        CHROMA_PATH = os.getenv("CHROMA_PATH")

        command = f"nohup chroma run --host 0.0.0.0 --port {CHROMA_PORT} --path {CHROMA_PATH} > chroma.log 2>&1 &"
        subprocess.Popen(command, shell=True)
        print("ChromaDB가 백그라운드에서 실행되었습니다.")

    except FileNotFoundError:
        print("❌ 'chroma' 명령어를 찾을 수 없습니다. PATH에 등록되어 있는지 확인하세요.")
    except Exception as e:
        print("⚠️ 예기치 못한 오류 발생:", e)

if __name__ == "__main__":
    run_chroma()
