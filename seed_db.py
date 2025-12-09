# seed_db.py
import os, shutil
from sqlalchemy import create_engine, Column, Integer, String, Boolean, UniqueConstraint
from sqlalchemy.orm import declarative_base

SEED_NAME = "eers_data_seed.db"

Base = declarative_base()

class Notice(Base):
    __tablename__ = 'notices'
    id = Column(Integer, primary_key=True, autoincrement=True)
    is_favorite   = Column(Boolean, default=False, index=True)
    stage         = Column(String)
    biz_type      = Column(String)
    project_name  = Column(String)
    client        = Column(String)
    address       = Column(String)
    phone_number  = Column(String)
    model_name    = Column(String, nullable=False, default='N/A')
    quantity      = Column(Integer)
    amount        = Column(String)
    is_certified  = Column(String)
    notice_date   = Column(String, index=True)
    detail_link   = Column(String, nullable=False)
    assigned_office = Column(String, index=True)
    status        = Column(String, default='')
    memo          = Column(String, default='')
    __table_args__ = (
        UniqueConstraint('detail_link', 'model_name', name='_detail_model_uc'),
    )

def build_empty_seed():
    if os.path.exists(SEED_NAME):
        os.remove(SEED_NAME)
    engine = create_engine(f"sqlite:///{SEED_NAME}", future=True)
    Base.metadata.create_all(engine)
    print(f"[OK] created empty seed: {SEED_NAME}")

def copy_from_existing(src_path: str):
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    shutil.copy(src_path, SEED_NAME)
    print(f"[OK] copied seed from: {src_path} -> {SEED_NAME}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-db", help="기존 eers_data.db 경로에서 씨드 복사")
    args = ap.parse_args()
    if args.from_db:
        copy_from_existing(args.from_db)
    else:
        build_empty_seed()
