## chroma db에 업데이트 잘 됐는지 확인 및 json으로 저장
## python db/chroma_check.py

import os
import json
from chroma_client import (
    get_user_latest_collection,
    get_user_history_collection,
)

# MBTI 컬렉션
latest_mbti = get_user_latest_collection()
history_mbti = get_user_history_collection()


print("=== USER Latest ===")
latest_raw = latest_mbti.peek(limit=None)
print(latest_raw)


print("\n=== USER History ===")
history_raw = history_mbti.peek(limit=None)
print(history_raw)


# peek 결과를 항목 리스트로 변환
def to_items(raw):
    ids = raw.get('ids', [])
    docs = raw.get('documents', [])
    embs = raw.get('embeddings', [])
    metas = raw.get('metadatas', [])
    items = []
    for i, _id in enumerate(ids):
        emb = embs[i] if i < len(embs) else None
        try:
            emb = emb.tolist()
        except Exception:
            pass
        items.append({
            'id': _id,
            'document': docs[i] if i < len(docs) else None,
            'embedding': emb,
            'metadata': metas[i] if i < len(metas) else {},
        })
    return items

# latest_items = to_items(latest_raw)
history_items = to_items(history_raw)

# user_id별 항목 그룹화 
data_by_user = {}
for item in history_items:
    user_id = item['metadata'].get('user_id')
    if not user_id:
        continue
    data_by_user.setdefault(user_id, []).append(item)
    

# 파일 저장
output_dir = 'mbti_user'
os.makedirs(output_dir, exist_ok=True)

for user_id, entries in data_by_user.items():
    filename=os.path.join(output_dir, f"mbti_user_{user_id}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print("save file")