import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
save_path = "models/hyperclovax-1.5b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
model.save_pretrained(save_path)
