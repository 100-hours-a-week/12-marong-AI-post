import os
import uuid
import argparse
import numpy as np
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

from mbti_update import update_mbti
from chroma_client import (
    get_chroma_client,
    get_user_latest_collection,
    get_user_history_collection,
)

# .env 파일 로드
load_dotenv()

def load_config():
    return {
        "db": {
            "host":     os.environ["DB_HOST"],
            "user":     os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
            "database": os.environ["DB_NAME"],
            "charset":  os.environ.get("DB_CHARSET", "utf8mb4")
        },
        "change_weight": int(os.environ.get("CHANGE_WEIGHT", 5)),
    }

class MBTIUpdateService:
    def __init__(self, config):
        self.change_weight = config["change_weight"]
        # MySQL 연결
        self.mysql_conn = mysql.connector.connect(**config["db"])
        self.cursor = self.mysql_conn.cursor(dictionary=True)
        # ChromaDB 연결
        get_chroma_client()
        self.user_latest_col   = get_user_latest_collection()
        self.user_history_col  = get_user_history_collection()


    # Users 테이블의 모든 id 조회
    def fetch_all_users(self) -> list[int]:
        self.cursor.execute("SELECT id AS user_id FROM Users")
        return [r["user_id"] for r in self.cursor.fetchall()]

    # 모든 user_id에 대한 feed 조회
    def fetch_feed(self, user_id: int) -> int:
        self.cursor.execute(
            "SELECT content FROM Posts WHERE user_id = %s", (user_id,)
        )
        rows = self.cursor.fetchall()
        return " ".join(r["content"] for r in rows) if rows else ""
    

    def fetch_prev_data(self, user_id: int) -> tuple[dict, int | None, bool]:
        # ChromaDB 최신값 조회
        recs = self.user_latest_col.get(
            where={"user_id": int(user_id)}, limit=1, include=["metadatas"]
        )
        if recs.get("ids"):
            meta = recs["metadatas"][0]
            prev_scores = {
                "ei_score": meta["ei_score"],
                "sn_score": meta["sn_score"],
                "tf_score": meta["tf_score"],
                "jp_score": meta["jp_score"]
            }
            prev_hobby = meta.get("hobby_name") 
            return prev_scores, prev_hobby, False

        # 신규 유저: 설문 결과 조회
        # MBTI 설문
        self.cursor.execute(
            "SELECT ei_score, sn_score, tf_score, jp_score"
            " FROM SurveyMBTI WHERE user_id = %s"
            " ORDER BY created_at DESC LIMIT 1", (user_id,)
        )
        mbti_row = self.cursor.fetchone()
        prev_scores = {
            "ei_score": mbti_row["ei_score"],
            "sn_score": mbti_row["sn_score"],
            "tf_score": mbti_row["tf_score"],
            "jp_score": mbti_row["jp_score"]
        }
        # 취미 설문
        self.cursor.execute(
            "SELECT hobby_name FROM SurveyHobby"
            " WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,)
        )
        hobby_row = self.cursor.fetchone()
        prev_hobby = hobby_row["hobby_name"] if hobby_row else None
        return prev_scores, prev_hobby, True

    # 가져오기 
    def run(self, user_id: int):
        feed = self.fetch_feed(user_id)
        if not feed:
            print(f"사용자 [{user_id}]의 피드가 존재하지 않습니다.")
        else:
            prev_scores, prev_hobby, is_new = self.fetch_prev_data(user_id)
            hobby_meta = prev_hobby if isinstance(prev_hobby, list) else (prev_hobby or [])
            updated = update_mbti(feed, prev_scores, self.change_weight)
            ts = datetime.now().isoformat()

            # 메타데이터 담기
            common_meta = {
                "user_id": int(user_id),
                "timestamp": ts,
                "hobby_name": hobby_meta,
                "ei_score": int(updated["mbti"]["ei_score"]),
                "sn_score": int(updated["mbti"]["sn_score"]),
                "tf_score": int(updated["mbti"]["tf_score"]),
                "jp_score": int(updated["mbti"]["jp_score"])
            }

            # embeddings에는 [0.0] 하나만 넘기기
            holder_vec=np.array([0.0], dtype=float)

            # user 히스토리 기록
            self.user_history_col.add(
                ids=[str(uuid.uuid4())],
                metadatas=[common_meta],
                documents=[""],
                embeddings=[holder_vec]
            )

        # user 최신값
        latest_meta = {**common_meta, "updated_at": ts }
        self.user_latest_col.upsert(
            ids=[str(user_id)],
            metadatas=[latest_meta],
            documents=[""],
            embeddings=[holder_vec]
        )


        # 결과 출력
        print(f"\n사용자 [{user_id}]의 MBTI가 업데이트되었습니다.")
        print(f"  MBTI 점수: {updated['mbti']}")
        print(f"  취미: {prev_hobby or '없음'}")


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(description="MBTI 업데이트 서비스")
    parser.add_argument(
        "user_id",
        nargs='?',  # Optional positional
        help="사용자 ID (지정하지 않으면 모든 유저 대상)"
    )
    parser.add_argument(
        "--weight", "-w",
        type=int,
        default=config["change_weight"],
        help="변동 가중치 (환경변수 CHANGE_WEIGHT)"
    )
    args = parser.parse_args()

    service = MBTIUpdateService(config)
    service.change_weight = args.weight

    if args.user_id:
        service.run(args.user_id)
    else:
        for uid in service.fetch_all_users():
            service.run(uid)
