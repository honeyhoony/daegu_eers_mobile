from __future__ import annotations
import os
import re
import logging
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, UniqueConstraint,
    DateTime, text, Table, MetaData
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
# =========================================================
# 로거 설정
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# Base 선언
# =========================================================
Base = declarative_base()


# =========================================================
# 모델 정의
# =========================================================
class Notice(Base):
    __tablename__ = "notices"

    id              = Column(Integer, primary_key=True)
    is_favorite     = Column(Boolean, default=False, nullable=False)
    stage           = Column(String)
    biz_type        = Column(String)
    project_name    = Column(String)
    client          = Column(String)
    address         = Column(String)
    phone_number    = Column(String)
    model_name      = Column(String, default="N/A")
    quantity        = Column(Integer)
    amount          = Column(String)
    is_certified    = Column(String)
    notice_date     = Column(String)
    detail_link     = Column(String, nullable=False)
    assigned_office = Column(String, default="관할지사확인요망")
    status          = Column(String, default="")
    memo            = Column(String, default="")
    source_system   = Column(String, default="G2B", nullable=False)
    kapt_code       = Column(String)

    __table_args__ = (
        UniqueConstraint(
            "source_system", "detail_link", "model_name", "assigned_office",
            name="uq_notice_unique"
        ),
    )


class MailRecipient(Base):
    __tablename__ = "mail_recipients"

    id        = Column(Integer, primary_key=True)
    office    = Column(String, nullable=False)
    email     = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    name      = Column(String)

    __table_args__ = (
        UniqueConstraint("office", "email", name="uq_mail_recipient"),
    )


class MailHistory(Base):
    __tablename__ = "mail_history"

    id           = Column(Integer, primary_key=True)
    sent_at      = Column(DateTime, default=datetime.utcnow)
    office       = Column(String, nullable=False)
    subject      = Column(String, nullable=False)
    period_start = Column(String, nullable=False)
    period_end   = Column(String, nullable=False)
    to_list      = Column(String, nullable=False)
    cc_list      = Column(String, default="")
    total_count  = Column(Integer, default=0)
    attach_name  = Column(String, default="")
    preview_html = Column(String, default="")


# =========================================================
# DB URL 로딩
# =========================================================
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL", "")
if not DATABASE_URL:
    logger.warning("⚠️ SUPABASE_DATABASE_URL 환경변수 미설정 — SQLite로 fallback.")
    DATABASE_URL = "sqlite:///eers.db"
else:
    DATABASE_URL = re.sub(r"[?&]pgbouncer=true", "", DATABASE_URL)
    logger.info(f"✅ Using Supabase DB: {DATABASE_URL[:50]}...")

# =========================================================
# Engine & Session 생성
# =========================================================
try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.warning(f"⚠️ 테이블 생성 중 오류 발생 (무시됨): {e}")
except Exception as e:
    logger.exception(f"❌ DB 연결 실패: {e}")
    raise

# =========================================================
# 세션 제공 함수
# =========================================================
def get_db_session():
    return SessionLocal()

# =========================================================
# KEA 캐시 관련 함수
# =========================================================
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

    return int(row[0]) if row else None


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
            "f": int(flag),
            "ts": datetime.utcnow().isoformat(timespec="seconds")
        }
    )


# database.py (간단 버전)
from sqlalchemy import Table, Column, String, MetaData
# =========================================================
# 메타 테이블 관리
# =========================================================
meta = MetaData()
MetaKV = Table(
    "meta_kv", meta,
    Column("k", String, primary_key=True),
    Column("v", String),
)
Base.metadata.create_all(bind=engine)

def set_meta(session, k, v):
    session.execute(
        text("""INSERT INTO meta_kv(k,v) VALUES(:k,:v)
                ON CONFLICT (k) DO UPDATE SET v = excluded.v"""),
        {"k": k, "v": v}
    )

def get_meta(session, k, default=None):
    row = session.execute(text("SELECT v FROM meta_kv WHERE k=:k"), {"k": k}).fetchone()
    return row[0] if row else default


# ===============================
# UPSERT 헬퍼 함수 (중복 제거 포함)
# ===============================
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

# =========================================================
# UPSERT 관련 유틸
# =========================================================
def get_dialect_insert(engine):
    dialect = engine.url.get_backend_name()
    return sqlite_insert if "sqlite" in dialect else pg_insert

def dedupe_by_unique_key(rows):
    seen = {}
    for r in rows:
        key = (r["source_system"], r["detail_link"], r["model_name"], r["assigned_office"])
        seen[key] = r
    return list(seen.values())

def bulk_upsert_notices(session, rows):
    if not rows:
        return
    rows = dedupe_by_unique_key(rows)
    insert_stmt = get_dialect_insert(engine)(Notice).values(rows)
    insert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["source_system", "detail_link", "model_name", "assigned_office"],
        set_={col.name: insert_stmt.excluded[col.name] for col in Notice.__table__.columns if col.name not in ("id",)}
    )
    session.execute(insert_stmt)
    session.commit()
