from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, UniqueConstraint,
    DateTime, text, MetaData, Table
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert


def show_db_debug_panel():
    st.sidebar.markdown("### 🧪 DB 연결 상태")

    try:
        session = get_db_session()
        try:
            db_name = session.execute(text("select current_database()")).scalar()
            now_db  = session.execute(text("select now()")).scalar()
            cnt     = session.query(Notice).count()

            st.sidebar.success("DB 연결 OK")
            st.sidebar.write(f"- DB: {db_name}")
            st.sidebar.write(f"- DB 시간: {now_db}")
            st.sidebar.write(f"- notices 건수: {cnt}")
        finally:
            session.close()

    except Exception as e:
        st.sidebar.error("DB 연결 실패")
        st.sidebar.code(str(e))



# =========================================================
# 로거
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# DB URL (Supabase ONLY)
# =========================================================
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ SUPABASE_DATABASE_URL is required (SQLite fallback disabled)")

# pgbouncer 옵션 제거 (SQLAlchemy 충돌 방지)
DATABASE_URL = re.sub(r"[?&]pgbouncer=true", "", DATABASE_URL)

logger.info(f"✅ Using Supabase DB (PostgreSQL)")

# =========================================================
# Engine / Session
# =========================================================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# =========================================================
# Base
# =========================================================
Base = declarative_base()

# =========================================================
# Models
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
# Meta KV (마지막 동기화 시각 등)
# =========================================================
meta = MetaData()

MetaKV = Table(
    "meta_kv", meta,
    Column("k", String, primary_key=True),
    Column("v", String),
)

# =========================================================
# Table 생성
# =========================================================
Base.metadata.create_all(bind=engine)
meta.create_all(bind=engine)

# =========================================================
# Session Helper
# =========================================================
def get_db_session():
    return SessionLocal()

# =========================================================
# Meta Helpers
# =========================================================
def set_meta(session, k: str, v: str):
    session.execute(
        text("""
            INSERT INTO meta_kv(k, v)
            VALUES (:k, :v)
            ON CONFLICT (k) DO UPDATE SET v = excluded.v
        """),
        {"k": k, "v": v}
    )
    session.commit()


def get_meta(session, k: str, default=None):
    row = session.execute(
        text("SELECT v FROM meta_kv WHERE k=:k"),
        {"k": k}
    ).fetchone()
    return row[0] if row else default

# =========================================================
# UPSERT (핵심)
# =========================================================
def dedupe_by_unique_key(rows):
    seen = {}
    for r in rows:
        key = (r["source_system"], r["detail_link"], r["model_name"], r["assigned_office"])
        seen[key] = r
    return list(seen.values())


def bulk_upsert_notices(session, rows):
    if not rows:
        logger.warning("[DB] No rows to upsert")
        return

    rows = dedupe_by_unique_key(rows)

    stmt = pg_insert(Notice).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_system", "detail_link", "model_name", "assigned_office"],
        set_={
            col.name: stmt.excluded[col.name]
            for col in Notice.__table__.columns
            if col.name != "id"
        }
    )

    session.execute(stmt)
    session.commit()

    # 🔴 운영 핵심 로그 (이게 Fly logs에 반드시 보여야 정상)
    total = session.query(Notice).count()
    logger.info(
        f"[DB OK] upsert={len(rows)}, total_notices={total}, db=postgresql"
    )
