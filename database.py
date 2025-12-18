from __future__ import annotations
import os
import re
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, UniqueConstraint,
    DateTime, text
)
from sqlalchemy.orm import declarative_base, sessionmaker


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
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DATABASE_URL 환경변수 누락됨")

DATABASE_URL = re.sub(r"[?&]pgbouncer=true", "", DATABASE_URL)


# =========================================================
# Engine & Session 생성
# =========================================================
engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"},
    pool_size=5,
    max_overflow=0,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# 테이블 생성
Base.metadata.create_all(bind=engine)


# =========================================================
# 세션 제공 함수 (app.py에서 import)
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

def get_dialect_insert(engine):
    """DB 종류에 따라 적절한 insert 함수 반환"""
    dialect = engine.url.get_backend_name()
    if "sqlite" in dialect:
        return sqlite_insert
    elif "postgres" in dialect:
        return pg_insert
    else:
        raise RuntimeError(f"Unsupported DB dialect: {dialect}")

def dedupe_by_unique_key(rows):
    """source_system + detail_link + model_name + assigned_office 중복 제거"""
    seen = {}
    for r in rows:
        key = (
            r["source_system"],
            r["detail_link"],
            r["model_name"],
            r["assigned_office"],
        )
        # 마지막 값으로 덮어쓰기 (최신 정보 유지)
        seen[key] = r
    return list(seen.values())

def bulk_upsert_notices(session, rows):
    """중복 제거 후 안전하게 UPSERT"""
    if not rows:
        return

    # ✅ [중복 제거 추가]
    rows = dedupe_by_unique_key(rows)

    insert_stmt = pg_insert(Notice).values(rows)
    insert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["source_system", "detail_link", "model_name", "assigned_office"],
        set_={
            "stage": insert_stmt.excluded.stage,
            "biz_type": insert_stmt.excluded.biz_type,
            "project_name": insert_stmt.excluded.project_name,
            "client": insert_stmt.excluded.client,
            "address": insert_stmt.excluded.address,
            "phone_number": insert_stmt.excluded.phone_number,
            "model_name": insert_stmt.excluded.model_name,
            "quantity": insert_stmt.excluded.quantity,
            "amount": insert_stmt.excluded.amount,
            "is_certified": insert_stmt.excluded.is_certified,
            "notice_date": insert_stmt.excluded.notice_date,
            "detail_link": insert_stmt.excluded.detail_link,
            "assigned_office": insert_stmt.excluded.assigned_office,
            "source_system": insert_stmt.excluded.source_system,
            "kapt_code": insert_stmt.excluded.kapt_code,
        },
    )

    session.execute(insert_stmt)
    session.commit()
