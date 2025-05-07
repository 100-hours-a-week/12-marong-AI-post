import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Load .env token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 모델 이름 & 저장 경로
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
save_path = "models/hyperclovax-1.5b-instruct"

# 디렉토리 없으면 생성
os.makedirs(save_path, exist_ok=True)

# Config 다운로드 및 저장
config = AutoConfig.from_pretrained(model_name, token=hf_token)
config.save_pretrained(save_path)

# Tokenizer 다운로드 및 저장
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.save_pretrained(save_path)

# Model 다운로드 및 저장
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
model.save_pretrained(save_path)

# print(f"모델과 토크나이저가 '{save_path}'에 저장되었습니다")