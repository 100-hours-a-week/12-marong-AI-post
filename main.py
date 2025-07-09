import os
import uuid
import argparse
import numpy as np
from typing import Tuple, List
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from db.db import SessionLocal
from db.db_model import Users, Posts, SurveyMBTI, Missions, MbtiUpdates
from core.update_mbti import MBTIUpdater

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

    # Users 테이블의 모든 id 조회
    def fetch_all_users(self) -> list[int]:
        return [u.id for u in self.db.query(Users.id).all()]

    # 모든 user_id에 대한 feed 조회
    def fetch_feed(self, user_id: int) -> Tuple[str, List[str]]:
        rows = self.db.query(Posts.content, Missions.title).join(Missions, Posts.mission_id == Missions.id).filter(Posts.user_id == user_id).all()
        feed_texts = [f"[{m}] {p}" for p, m in rows]
        mission_titles = [m for _, m in rows]
        return " ".join(feed_texts), mission_titles
    
    def fetch_prev_data(self, user_id: int) -> dict:
        latest_update = (
            self.db.query(MbtiUpdates)
            .filter(MbtiUpdates.user_id == user_id)
            .order_by(MbtiUpdates.created_at.desc())
            .first()
        )
        if latest_update:
            return {
                "ei_score": latest_update.ei_score,
                "sn_score": latest_update.sn_score,
                "tf_score": latest_update.tf_score,
                "jp_score": latest_update.jp_score
            }

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
        return prev_scores

    # 가져오기 
    def run(self, user_id: int):
        rows = self.db.query(Posts.id, Posts.content, Missions.title).join(Missions, Posts.mission_id == Missions.id).filter(Posts.user_id == user_id).all()
        if not rows:
            print(f"사용자 [{user_id}]의 피드가 존재하지 않습니다.")
            return
        
        prev_scores = self.fetch_prev_data(user_id)
        original_scores = prev_scores.copy()
        current_scores = prev_scores.copy()
    

        for idx, (post_id, post_content, mission_title) in enumerate(rows, 1):
            user_feed = f"[{mission_title}] {post_content}"
            mission_text = mission_title
            print(f"\n [{idx}] 미션: {mission_text}")

            # if len(post_content.strip()) <= 5:
            #     print(f" {post_content} 피드 내용이 너무 짧아 MBTI를 판단할 수 없습니다")
            #     continue
            prev_scores_before = prev_scores.copy()
            mission_based_feed = f"유저는 '{mission_text}' 미션을 수행하며 아래 피드를 작성했습니다: \n{user_feed}"
            updated = self.updater.update_mbti(mission_based_feed, prev_scores, mission_text, original_scores)
            current_scores =updated["mbti"]
            final_reason = updated["final_reason"]
            changed_axis = updated["changed_axis"].upper()[:2]
            prev_scores = updated["original_score"]

            # 결과 출력
            print(f"\n사용자 [{user_id}]의 MBTI가 업데이트되었습니다.")
            print(f"MBTI 점수: {updated['mbti']}")
            print(f"이유: {updated['final_reason']}")

            # 축별 키 만들기
            axis_key = f"{changed_axis}_score" 

            # 점수만 추출
            axis_key = axis_key.lower()
            previous_score_value = prev_scores_before[axis_key]
            current_score_value = current_scores[axis_key]

            prev_scores = updated["mbti"]


            update_record = MbtiUpdates(
                post_id=post_id,
                user_id=user_id,
                ei_score=current_scores["ei_score"],
                sn_score=current_scores["sn_score"],
                tf_score=current_scores["tf_score"],
                jp_score=current_scores["jp_score"],
                changed_mbti_type=changed_axis,
                change_reason=final_reason,
                previous_score=previous_score_value,
                current_score=current_score_value,
                created_at=datetime.now()
            )

            self.db.add(update_record)
            self.db.commit()
            prev_scores = current_scores.copy()
            print(f"\n 사용자 [{user_id}]의 최종 MBTI 결과가 테이블에 저장되었습니다.")

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