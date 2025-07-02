import os
import uuid
import argparse
import numpy as np
from typing import Tuple, List
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from db.db import SessionLocal
from db.db_model import Users, Posts, SurveyMBTI, SurveyHobby, Missions
from core.update_mbti import MBTIUpdater
from db.chroma_client import (
    get_chroma_client,
    get_user_latest_collection,
    get_user_history_collection,
)

# .env 파일 로드
load_dotenv()

def load_config():
    return {
        "change_weight": int(os.environ.get("CHANGE_WEIGHT")),
    }

class MBTIUpdateService:
    def __init__(self, config):
        self.change_weight = config["change_weight"]
        # MBTIUpdater
        self.updater = MBTIUpdater(change_weight = self.change_weight)

        # MySQL 연결
        self.db: Session = SessionLocal()

        # ChromaDB 연결
        get_chroma_client()
        self.user_latest_col   = get_user_latest_collection()
        self.user_history_col  = get_user_history_collection()


    # Users 테이블의 모든 id 조회
    def fetch_all_users(self) -> list[int]:
        return [u.id for u in self.db.query(Users.id).all()]

    # 모든 user_id에 대한 feed 조회
    def fetch_feed(self, user_id: int) -> Tuple[str, List[str]]:
        rows = self.db.query(Posts.content, Missions.title).join(Missions, Posts.mission_id == Missions.id).filter(Posts.user_id == user_id).all()
        feed_texts = [f"[{m}] {p}" for p, m in rows]
        mission_titles = [m for _, m in rows]
        return " ".join(feed_texts), mission_titles
    
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
        mbti_row = (
            self.db.query(SurveyMBTI)
            .filter(SurveyMBTI.user_id == user_id)
            .order_by(SurveyMBTI.created_at.desc())
            .first()
        )
        if not mbti_row:
            raise ValueError(f"사용자 {user_id}의 MBTI 설문 결과가 없습니다.")
        prev_scores = {
            "ei_score": mbti_row.ei_score,
            "sn_score": mbti_row.sn_score,
            "tf_score": mbti_row.tf_score,
            "jp_score": mbti_row.jp_score
        }
        hobby_row = (
            self.db.query(SurveyHobby)
            .filter(SurveyHobby.user_id == user_id)
            .order_by(SurveyHobby.created_at.desc())
            .first()
        )
        prev_hobby = hobby_row.hobby_name if hobby_row else None

        return prev_scores, prev_hobby, True

    # 가져오기 
    def run(self, user_id: int):
        rows = self.db.query(Posts.content, Missions.title).join(Missions, Posts.mission_id == Missions.id).filter(Posts.user_id == user_id).all()
        if not rows:
            print(f"사용자 [{user_id}]의 피드가 존재하지 않습니다.")
            return
        
        prev_scores, prev_hobby, is_new = self.fetch_prev_data(user_id)
        hobby_meta = prev_hobby or []
        ts = datetime.now().isoformat()

        for post_content, mission_title in rows:
            user_feed = f"[{mission_title}] {post_content}"
            mission_text = mission_title

            if len(post_content.strip()) <= 5:
                print(f" {post_content} 피드 내용이 너무 짧아 MBTI를 판단할 수 없습니다")
                updated = {
                    "mbti": prev_scores,
                    "final_reason": "피드 내용이 너무 짧아 MBTI를 판단할 수 없습니다."
                }
            else:
                mission_based_feed = f"유저는 '{mission_text}' 미션을 수행하며 아래 피드를 작성했습니다: \n{user_feed}"
                updated = self.updater.update_mbti(mission_based_feed, prev_scores, mission_text)

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
        print(f"MBTI 점수: {updated['mbti']}")
        print(f"취미: {prev_hobby or '없음'}")
        print(f"이유: {updated['final_reason']}")


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