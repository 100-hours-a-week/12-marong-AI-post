from sqlalchemy import (
    Column, String, Integer, BigInteger, Text, Boolean, Date,
    ForeignKey, TIMESTAMP, UniqueConstraint, Index, CHAR
)
from sqlalchemy import Index, desc
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from .db import Base


class Users(Base):
    __tablename__ = "Users"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    email = Column(String(100), unique=True, nullable=False)
    provider_id = Column(String(100), unique=True, nullable=False)
    nickname = Column(String(200), nullable=False)
    provider_name = Column(String(100))
    profile_image_url = Column(Text)
    status = Column(String(40), default='active')
    has_completed_survey = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP)


class Groups(Base):
    __tablename__ = "Groups"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    invite_code = Column(CHAR(6), unique=True, nullable=False)
    image_url = Column(Text)


class UserGroups(Base):
    __tablename__ = "UserGroups"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('Users.id', ondelete='CASCADE'), nullable=False)
    group_id = Column(BigInteger, ForeignKey('Groups.id', ondelete='CASCADE'), nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'group_id', name='uq_user_group'),)


class Missions(Base):
    __tablename__ = "Missions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    difficulty = Column(String(10), nullable=False)


class Posts(Base):
    __tablename__ = "Posts"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('Users.id', ondelete='CASCADE'), nullable=False)
    group_id = Column(BigInteger, ForeignKey('Groups.id', ondelete='CASCADE'), nullable=False)
    week = Column(Integer, nullable=False)
    mission_id = Column(BigInteger, ForeignKey('Missions.id', ondelete='CASCADE'), nullable=False)
    anonymous_snapshot_name = Column(String(200), nullable=False)
    manittee_name = Column(String(200))
    content = Column(Text, nullable=False)
    image_url = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP)

    __table_args__ = (
        Index('idx_group_created_at', 'group_id', desc('created_at')),
        Index('idx_week_created_at', 'week', desc('created_at')),
    )


class GroupMissions(Base):
    __tablename__ = "GroupMissions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    group_id = Column(BigInteger, ForeignKey('Groups.id', ondelete='CASCADE'), nullable=False)
    mission_id = Column(BigInteger, ForeignKey('Missions.id', ondelete='CASCADE'), nullable=False)
    max_assignable = Column(Integer, default=0)
    remaining_count = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint('group_id', 'mission_id', name='uq_group_mission'),)


class UserMissions(Base):
    __tablename__ = "UserMissions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('Users.id', ondelete='CASCADE'), nullable=False)
    group_id = Column(BigInteger, ForeignKey('Groups.id', ondelete='CASCADE'), nullable=False)
    mission_id = Column(BigInteger, ForeignKey('Missions.id', ondelete='CASCADE'), nullable=False)
    status = Column(String(20), default='ing')
    week = Column(Integer, nullable=False)
    assigned_date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_user_group_week', 'user_id', 'group_id', 'week'),
        Index('idx_user_assigned_date', 'user_id', 'assigned_date'),
    )


class SurveyMBTI(Base):
    __tablename__ = "SurveyMBTI"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('Users.id', ondelete='CASCADE'), nullable=False)
    ei_score = Column(Integer, nullable=False)
    sn_score = Column(Integer, nullable=False)
    tf_score = Column(Integer, nullable=False)
    jp_score = Column(Integer, nullable=False)
    mbti = Column(String(10), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class SurveyHobby(Base):
    __tablename__ = "SurveyHobby"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('Users.id', ondelete='CASCADE'), nullable=False)
    hobby_name = Column(String(100), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    invite_code = Column(CHAR(6), unique=True, nullable=False)
    image_url = Column(Text)