import os
import uuid
import argparse
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

from mbti_update import update_mbti
from chroma_client_db import (
    get_chroma_client,
    get_mbti_latest_collection,
    get_mbti_history_collection,
    get_hobby_latest_collection,
    get_hobby_history_collection
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
        self.mbti_latest_col   = get_mbti_latest_collection()
        self.mbti_history_col  = get_mbti_history_collection()
        self.hobby_latest_col  = get_hobby_latest_collection()
        self.hobby_history_col = get_hobby_history_collection()

    def fetch_all_users(self) -> list[str]:
        """
        Users 테이블의 모든 id 조회
        """
        self.cursor.execute("SELECT id AS user_id FROM Users")
        return [r["user_id"] for r in self.cursor.fetchall()]

    def fetch_feed(self, user_id: str) -> str:
        self.cursor.execute(
            "SELECT content FROM Posts WHERE user_id = %s", (user_id,)
        )
        rows = self.cursor.fetchall()
        return " ".join(r["content"] for r in rows) if rows else ""

    def fetch_prev_scores(self, user_id: str) -> tuple[dict, bool]:
        # 1) ChromaDB 최신값 조회 → 있으면 기존 유저
        recs = self.mbti_latest_col.get(
            where={"user_id": str(user_id)}, limit=1, include=["embeddings"]
        )
        if recs.get("ids"):
            vec = recs["embeddings"][0]
            return dict(zip(["E", "N", "F", "P"], vec)), False
        # 2) ChromaDB에 없으면 신규 유저: SurveyMBTI 설문 결과 조회
        self.cursor.execute(
            """
            SELECT ei_score, sn_score, tf_score, jp_score
            FROM SurveyMBTI
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        )
        row = self.cursor.fetchone()
        return {
            "E": row["ei_score"],
            "N": row["sn_score"],
            "F": row["tf_score"],
            "P": row["jp_score"]
        }, True

    def fetch_prev_hobby(self, user_id: str) -> tuple[str | None, bool]:
        # 1) ChromaDB 최신값 조회 → 있으면 기존 유저
        recs = self.hobby_latest_col.get(
            where={"user_id": str(user_id)}, limit=1, include=["metadatas"]
        )
        if recs.get("ids"):
            return recs["metadatas"][0].get("hobby_name"), False
        # 2) ChromaDB에 없으면 신규 유저: SurveyHobby 설문 결과 조회
        self.cursor.execute(
            "SELECT hobby_name FROM SurveyHobby WHERE user_id = %s", (user_id,)
        )
        row = self.cursor.fetchone()
        return row["hobby_name"], True

    def run(self, user_id: str):
        feed = self.fetch_feed(user_id)
        if not feed:
            print(f"사용자 [{user_id}]의 피드가 존재하지 않습니다.")
            return

        prev_scores, is_new_scores = self.fetch_prev_scores(user_id)
        prev_hobby,  is_new_hobby  = self.fetch_prev_hobby(user_id)
        is_new_user = is_new_scores and is_new_hobby

        # MBTI 업데이트
        updated = update_mbti(feed, prev_scores, self.change_weight)
        new_vec = [updated["mbti"][k] for k in ["E", "N", "F", "P"]]
        ts = datetime.now().isoformat()

        # MBTI 히스토리 기록
        self.mbti_history_col.add(
            ids=[str(uuid.uuid4())],
            metadatas=[{"user_id": str(user_id), "timestamp": ts, "hobby": prev_hobby or ""}],
            documents=[""],
            embeddings=[new_vec]
        )
        # MBTI 최신값 upsert
        self.mbti_latest_col.upsert(
            ids=[str(user_id)],
            metadatas=[{"user_id": str(user_id), "updated_at": ts, "hobby": prev_hobby}],
            documents=[""],
            embeddings=[new_vec]
        )

        # 취미 히스토리 및 최신값 기록
        if prev_hobby:
            self.hobby_history_col.add(
                ids=[str(uuid.uuid4())],
                metadatas=[{"user_id": str(user_id), "timestamp": ts, "hobby_name": prev_hobby}],
                documents=[""],
                embeddings=[[0.0]]
            )
            self.hobby_latest_col.upsert(
                ids=[str(user_id)],
                metadatas=[{"user_id": str(user_id), "hobby_name": prev_hobby, "updated_at": ts}],
                documents=[""],
                embeddings=[[0.0]]
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
