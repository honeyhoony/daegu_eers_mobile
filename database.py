from __future__ import annotations
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, UniqueConstraint,
    DateTime, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# ─────────────────────────────────────────
# 1) Base 선언
# ─────────────────────────────────────────
Base = declarative_base()

# ─────────────────────────────────────────
# 2) 모델 정의
# ─────────────────────────────────────────
class Notice(Base):
    __tablename__ = "notices"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    is_favorite     = Column(Boolean, default=False, index=True, nullable=False)
    stage           = Column(String)
    biz_type        = Column(String)
    project_name    = Column(String)
    client          = Column(String)
    address         = Column(String)
    phone_number    = Column(String)
    model_name      = Column(String, nullable=False, default="N/A")
    quantity        = Column(Integer)
    amount          = Column(String)
    is_certified    = Column(String)
    notice_date     = Column(String, index=True)
    detail_link     = Column(String, nullable=False)
    assigned_office = Column(String, nullable=False, index=True, default="관할지사확인요망")
    status          = Column(String, default="")
    memo            = Column(String, default="")
    source_system   = Column(String, default="G2B", index=True, nullable=False)
    kapt_code       = Column(String, index=True, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "source_system", "detail_link", "model_name", "assigned_office",
            name="_source_detail_model_office_uc"
        ),
    )


class MailRecipient(Base):
    __tablename__ = "mail_recipients"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    office    = Column(String, index=True, nullable=False)
    email     = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    name      = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("office", "email", name="uq_office_email"),
    )


class MailHistory(Base):
    __tablename__ = "mail_history"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    sent_at      = Column(DateTime, default=datetime.utcnow, index=True)
    office       = Column(String, index=True, nullable=False)
    subject      = Column(String, nullable=False)
    period_start = Column(String, nullable=False)
    period_end   = Column(String, nullable=False)
    to_list      = Column(String, nullable=False)
    cc_list      = Column(String, default="")
    total_count  = Column(Integer, default=0)
    attach_name  = Column(String, default="")
    preview_html = Column(String, default="")

# ─────────────────────────────────────────
# 3) 엔진 & 세션 생성 함수
# ─────────────────────────────────────────
def get_engine_and_session(db_url: str):
    """
    Supabase / PostgreSQL 연결용 엔진 & 세션 생성.
    db_url 예:
      - postgresql://postgres:비번@db.xxx.supabase.co:5432/postgres
      - postgresql://postgres.프로젝트id:비번@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres?pgbouncer=true&sslmode=require
    """

    if not db_url:
        raise ValueError("DB URL is not set.")

    # psycopg2 드라이버 사용
    # 수정안 (드라이버 자동 선택)
    if db_url.startswith("postgresql://") and "://" in db_url:
        # psycopg2-binary 설치가 안된 환경에서도 작동하도록 pg8000 허용
        if "+psycopg2" not in db_url:
            db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)

    # Supabase URL에 이미 ?sslmode=require 가 붙어 있으면 그대로 사용
    # 따로 connect_args 안 쓰고 URL만 넘김
    engine = create_engine(
        db_url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"},
    )

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    return engine, SessionLocal

# ─────────────────────────────────────────
# 4) 초기 엔진/세션 선언
# ─────────────────────────────────────────
engine = None
SessionLocal = None

# ─────────────────────────────────────────
# 5) DB 세션 컨텍스트 관리
# ─────────────────────────────────────────
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────
# 6) KEA 모델 캐시 테이블 유틸
# ─────────────────────────────────────────
def _ensure_kea_cache_table(session):
    try:
        session.execute(text("SELECT 1 FROM kea_model_cache LIMIT 1"))
    except Exception:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS kea_model_cache (
                model_name  TEXT PRIMARY KEY,
                exists_flag INTEGER NOT NULL,
                checked_at  TEXT NOT NULL
            )
        """))


def _kea_cache_get(session, model: str):
    if not model:
        return None

    _ensure_kea_cache_table(session)
    row = session.execute(
        text("SELECT exists_flag FROM kea_model_cache WHERE model_name = :m"),
        {"m": model}
    ).fetchone()

    if row is None:
        return None

    return int(row[0])


def _kea_cache_set(session, model: str, flag: int):
    _ensure_kea_cache_table(session)
    session.execute(
        text("""
        INSERT INTO kea_model_cache(model_name, exists_flag, checked_at)
        VALUES (:m, :f, :ts)
        ON CONFLICT(model_name) DO UPDATE SET
            exists_flag = excluded.exists_flag,
            checked_at  = excluded.checked_at
        """),
        {
            "m": model,
            "f": int(bool(flag)),
            "ts": datetime.utcnow().isoformat(timespec="seconds")
        }
    )
    
import os

db_url = os.getenv("SUPABASE_DATABASE_URL")
engine, SessionLocal = get_engine_and_session(db_url)

