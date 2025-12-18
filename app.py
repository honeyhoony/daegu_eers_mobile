import streamlit as st
import re
import os
import math
import time
import threading
import calendar
import logging
import pandas as pd

from io import BytesIO
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import or_, func, inspect

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import extra_streamlit_components as stx

import os
import streamlit as st
from sqlalchemy import text



st.set_page_config(
    page_title="EERS 업무 지원 시스템",
    layout="wide",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

def get_secret(key: str, default=None):
    """
    Fly.io: 환경변수
    로컬(Streamlit): st.secrets
    """
    if key in os.environ:
        return os.environ.get(key)
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ===== mobile dummy functions =====
def fetch_dlvr_header(req_no):
    return {}

def fetch_dlvr_detail(req_no):
    return []

def fetch_data_for_stage(*args, **kwargs):
    return None

STAGES_CONFIG = {
    "G2B": {"name": "G2B", "code": "g2b"},
    "KAPT": {"name": "K-APT", "code": "kapt"},
}

def fetch_kapt_basic_info(code):
    return {}

def fetch_kapt_maintenance_history(code):
    return []

def send_mail(**kwargs):
    return True

def build_subject(*args):
    return "모바일 조회"

def build_body_html(*args):
    return "<html><body>모바일 조회</body></html>", None, None, None


# =========================
# 내부 모듈 (유지)
# =========================
from database import (
    Base,
    Notice,
    get_db_session,
    engine,
)

from collect_data import (
    fetch_data_for_stage,
    STAGES_CONFIG,
    fetch_kapt_basic_info,
    fetch_kapt_maintenance_history,
)
from pandas.tseries.offsets import BusinessDay


# =========================================================
# 로깅
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# config.py 또는 streamlit secrets 로드
# =========================================================
try:
    import config as _local_config
except ModuleNotFoundError:
    _local_config = None


def _cfg(name, default=None):
    if _local_config is not None and hasattr(_local_config, name):
        return getattr(_local_config, name)
    try:
        return st.secrets[name]
    except Exception:
        return default


# =========================================================
# DB URL 결정 (Supabase 유지)
# =========================================================
SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL") or _cfg("SUPABASE_DATABASE_URL", "")
if not SUPABASE_DATABASE_URL:
    st.error("FATAL: SUPABASE_DATABASE_URL이 없습니다.")
    st.stop()

# =========================================================
# 인증번호(로그인) 정책
#  - 로그인 = '수동 데이터 수집(API 호출)' 권한만 부여
#  - 조회/다운로드/데이터현황은 비로그인 허용
# =========================================================
ACCESS_CODE = os.environ.get("ACCESS_CODE")
if not ACCESS_CODE:
    raise RuntimeError("ACCESS_CODE 환경변수가 설정되지 않았습니다.")
ACCESS_CODE = ACCESS_CODE.strip()



COOKIE_NAME = "eers_access"

# 최소 동기화일
from datetime import date as _date_cls
_min_sync_raw = _cfg("MIN_SYNC_DATE", _date_cls(2025, 12, 1))
MIN_SYNC_DATE = (
    _date_cls.fromisoformat(_min_sync_raw)
    if isinstance(_min_sync_raw, str)
    else _min_sync_raw
)

SIX_MONTHS = timedelta(days=180)

# =========================================================
# 0-A. 공통 유틸
# =========================================================
# app.py (발췌) — last sync get/set 구현
from database import get_db_session
def _get_last_sync_datetime_from_meta():
    s = get_db_session()
    try:
        v = s.execute(text("SELECT v FROM meta_kv WHERE k='last_sync_dt'")).fetchone()
        return datetime.fromisoformat(v[0]) if v else None
    except Exception:
        return None
    finally:
        s.close()

def _set_last_sync_datetime_to_meta(dt: datetime):
    s = get_db_session()
    try:
        s.execute(text("""INSERT INTO meta_kv(k,v) VALUES('last_sync_dt', :v)
                          ON CONFLICT (k) DO UPDATE SET v = excluded.v"""),
                 {"v": dt.isoformat(timespec="seconds")})
        s.commit()
    finally:
        s.close()


# 사이드바 표시
last_dt = _get_last_sync_datetime_from_meta()
st.sidebar.info(
    f"자동수집: 08:00/12:00/19:00\n"
    f"마지막 수집: {last_dt or '기록 없음'}"
)

def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def prev_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while is_weekend(d):
        d -= timedelta(days=1)
    return d


def _as_date(val) -> Optional[date]:
    s = str(val or "").strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except ValueError:
            pass
    if len(s) == 10 and s.count("-") == 2:
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
    return None


def only_digits_gui(val):
    return re.sub(r"\D", "", str(val or ""))


def fmt_phone(val):
    v = only_digits_gui(val)
    if not v:
        return "정보 없음"
    if len(v) == 8:
        return f"{v[:4]}-{v[4:]}"
    if len(v) == 9:
        return f"{v[:2]}-{v[2:5]}-{v[5:]}"
    if len(v) == 10:
        return f"{v[:2]}-{v[2:6]}-{v[6:]}" if v.startswith("02") else f"{v[:3]}-{v[3:6]}-{v[6:]}"
    if len(v) == 11:
        return f"{v[:3]}-{v[3:7]}-{v[7:]}"
    return str(val)


# =========================================================
# 0-1. 상수
# =========================================================
OFFICES = [
    "전체", "직할", "동대구지사", "경주지사", "남대구지사", "서대구지사",
    "포항지사", "경산지사", "김천지사", "영천지사", "칠곡지사",
    "성주지사", "청도지사", "북포항지사", "고령지사", "영덕지사",
]
ITEMS_PER_PAGE = 100
DEFAULT_START_DATE = MIN_SYNC_DATE
DEFAULT_END_DATE = date.today()

CERT_TRUE_VALUES = {"O", "0", "Y", "YES", "1", "TRUE", "인증"}


def _normalize_cert(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip().upper()
    if not s:
        return ""
    if s in CERT_TRUE_VALUES:
        return "O"
    if s in {"X", "N", "NO", "미인증"}:
        return "X"
    return str(val)


def _fmt_int_commas(val):
    try:
        s = str(val or "").replace(",", "").strip()
        if not s or s.lower() == "none":
            return "정보 없음"
        n = int(float(s))
        return f"{n:,}"
    except Exception:
        return str(val) if val not in (None, "") else "정보 없음"


def _fmt_date_hyphen(val):
    s = str(val or "").strip()
    if not s:
        return "정보 없음"
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 6:
        y, m = digits[:4], digits[4:6]
        out = f"{y}-{m}"
        if len(digits) >= 8:
            d = digits[6:8]
            out = f"{out}-{d}"
        return out
    return s


def _fmt_phone_hyphen(val):
    v = re.sub(r"\D", "", str(val or ""))
    if not v:
        return "정보 없음"
    if len(v) == 8:
        return f"{v[:4]}-{v[4:]}"
    if len(v) == 9:
        return f"{v[:2]}-{v[2:5]}-{v[5:]}"
    if len(v) == 10:
        return f"{v[:2]}-{v[2:6]}-{v[6:]}" if v.startswith("02") else f"{v[:3]}-{v[3:6]}-{v[6:]}"
    if len(v) == 11:
        return f"{v[:3]}-{v[3:7]}-{v[7:]}"
    return str(val)


def _split_prdct_name(s: str):
    if not s: return "", "", ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    name = parts[0] if len(parts) >= 1 else s
    model = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "")
    spec = ", ".join(parts[3:]) if len(parts) >= 4 else ""
    return name, model, spec

def _pick(d: dict, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "-"): return v
    return default


def _to_int_local(val):
    try:
        return int(str(val).replace(",", "").strip() or 0)
    except Exception:
        return 0


# =========================================================
# 1) 쿠키/세션 기반 “수동 데이터수집 권한” (캡션형)
# =========================================================
def _cookie_manager():
    if "cookie_manager_instance" not in st.session_state:
        st.session_state["cookie_manager_instance"] = stx.CookieManager(key="eers_cookie_manager")
    return st.session_state["cookie_manager_instance"]


def has_sync_access() -> bool:
    if st.session_state.get("sync_access", False):
        return True

    cm = _cookie_manager()
    token = cm.get(cookie=COOKIE_NAME)

    if token == "1":
        st.session_state["sync_access"] = True
        return True

    return False



def grant_sync_access():
    # 🔒 이미 쿠키 세팅 중이면 재호출 차단
    if st.session_state.get("_setting_sync_cookie", False):
        return

    st.session_state["_setting_sync_cookie"] = True

    cm = _cookie_manager()
    st.session_state["sync_access"] = True

    expire_date = datetime.now() + timedelta(days=180)
    cm.set(COOKIE_NAME, "1", expires_at=expire_date)

    # 다음 rerun에서 다시 set 안 하도록
    st.session_state["_setting_sync_cookie_done"] = True


def revoke_sync_access():
    cm = _cookie_manager()
    try:
        cm.delete(cookie=COOKIE_NAME)
    except Exception:
        pass
    st.session_state["sync_access"] = False

def render_sidebar_sync_caption():
    st.sidebar.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # 이미 관리자면 아무것도 렌더하지 않음
    if has_sync_access():
        return

    st.session_state.setdefault("show_sync_code", False)

    # 아주 작은 캡션
    if st.sidebar.button("데이터 수집", key="admin_caption"):
        st.session_state["show_sync_code"] = True

    if st.session_state["show_sync_code"]:
        with st.sidebar.form("admin_auth_form"):
            code = st.text_input(
                "인증번호",
                type="password",
                placeholder="관리자 인증번호"
            )
            submitted = st.form_submit_button("확인")

        if submitted:
            input_code = code.strip().replace("\n", "").replace("\r", "")
            if input_code == ACCESS_CODE:
                grant_sync_access()   # ✅ 오직 여기서만 호출
                st.session_state["show_sync_code"] = False
                st.rerun()
            else:
                st.sidebar.error("인증번호가 올바르지 않습니다.")




# =========================================================
# 2) 세션 기본값
# =========================================================
def init_session_state():
    ss = st.session_state
    ss.setdefault("office", "전체")
    ss.setdefault("source", "전체")
    ss.setdefault("start_date", DEFAULT_START_DATE)
    ss.setdefault("end_date", DEFAULT_END_DATE)
    ss.setdefault("keyword", "")
    ss.setdefault("only_cert", False)
    ss.setdefault("include_unknown", False)

    ss.setdefault("page", 1)
    ss.setdefault("df_data", pd.DataFrame())
    ss.setdefault("total_items", 0)
    ss.setdefault("total_pages", 1)
    ss.setdefault("data_initialized", False)

    # 라우팅
    ss.setdefault("route_page", "공고 조회 및 검색")
    ss.setdefault("view_mode", "카드형")
    ss.setdefault("selected_notice", None)

    # sync 권한(로그인 대체)
    ss.setdefault("sync_access", False)


# =========================================================
# 3) 신규 건수 집계
# =========================================================
@st.cache_data(ttl=300)
def _get_new_item_counts_by_source_and_office() -> dict:
    session = get_db_session()
    if not session:
        return {}
    try:
        today = date.today()
        biz_today = today if not is_weekend(today) else prev_business_day(today)
        biz_prev = prev_business_day(biz_today)

        results = (
            session.query(
                Notice.assigned_office,
                Notice.source_system,
                func.count(Notice.id),
            )
            .filter(Notice.notice_date.in_([biz_today.isoformat(), biz_prev.isoformat()]))
            .group_by(Notice.assigned_office, Notice.source_system)
            .all()
        )

        counts = {}
        for office, source, count in results:
            office_name = office or ""
            if "/" in office_name:
                parts = [p.strip() for p in office_name.split("/") if p.strip()]
                for part in parts:
                    counts.setdefault(part, {"G2B": 0, "K-APT": 0})
                    source_key = "K-APT" if source == "K-APT" else "G2B"
                    counts[part][source_key] += count // max(1, len(parts))
            else:
                counts.setdefault(office_name, {"G2B": 0, "K-APT": 0})
                source_key = "K-APT" if source == "K-APT" else "G2B"
                counts[office_name][source_key] += count

        total_g2b = sum(v.get("G2B", 0) for v in counts.values())
        total_kapt = sum(v.get("K-APT", 0) for v in counts.values())
        counts["전체"] = {"G2B": total_g2b, "K-APT": total_kapt}
        return counts
    except Exception as e:
        logger.exception(f"신규 건수 집계 오류: {e}")
        return {}
    finally:
        session.close()



# =========================================================
# 4) 데이터 로딩 (공고 조회) - 비로그인 허용
# =========================================================
@st.cache_data(ttl=600, show_spinner="데이터를 조회 중...")
def load_data_from_db(
    office, source, start_date, end_date, keyword, only_cert, include_unknown, page,
):
    session = get_db_session()
    if not session:
        return pd.DataFrame(), 0

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    query = session.query(Notice).filter(
        Notice.notice_date.between(start_date_str, end_date_str)
    )

    if source == "나라장터":
        query = query.filter(Notice.source_system == "G2B")
    elif source == "K-APT":
        query = query.filter(Notice.source_system == "K-APT")

    if office and office != "전체":
        query = query.filter(
            or_(
                Notice.assigned_office == office,
                Notice.assigned_office.like(f"{office}/%"),
                Notice.assigned_office.like(f"%/{office}"),
                Notice.assigned_office.like(f"%/{office}/%"),
            )
        )

    if only_cert:
        query = query.filter(
            or_(
                Notice.is_certified == "O", Notice.is_certified == "0",
                Notice.is_certified == "Y", Notice.is_certified == "YES",
                Notice.is_certified == "1", Notice.is_certified == "인증"
            )
        )

    if not include_unknown:
        query = query.filter(
            ~Notice.assigned_office.like("%/%"),
            ~Notice.assigned_office.ilike("%불명%"),
            ~Notice.assigned_office.ilike("%미확인%"),
            ~Notice.assigned_office.ilike("%확인%"),
            ~Notice.assigned_office.ilike("%미정%"),
            ~Notice.assigned_office.ilike("%UNKNOWN%")
        )

    keyword_text = (keyword or "").strip()
    if keyword_text:
        cols = [Notice.project_name, Notice.client, Notice.model_name]
        terms = [t.strip() for t in keyword_text.split() if t.strip() and not t.startswith("-")]
        if terms:
            query = query.filter(or_(*[
                or_(*[c.ilike(f"%{term}%") for c in cols]) for term in terms
            ]))

    total_items = query.count()
    offset = (page - 1) * ITEMS_PER_PAGE
    rows = (
        query.order_by(Notice.notice_date.desc(), Notice.id.desc())
        .offset(offset)
        .limit(ITEMS_PER_PAGE)
        .all()
    )

    data = []
    today = date.today()
    biz_today = today if not is_weekend(today) else prev_business_day(today)
    biz_prev = prev_business_day(biz_today)
    new_days = {biz_today.isoformat(), biz_prev.isoformat()}

    for n in rows:
        is_new = n.notice_date in new_days
        phone_disp = fmt_phone(n.phone_number or "")
        cert_val = _normalize_cert(n.is_certified)

        data.append({
            "id": n.id,
            "구분": "K-APT" if n.source_system == "K-APT" else "나라장터",
            "사업소": (n.assigned_office or "").replace("/", "\n"),
            "단계": n.stage or "",
            "사업명": n.project_name or "",
            "기관명": n.client or "",
            "소재지": n.address or "",
            "연락처": phone_disp,
            "모델명": n.model_name or "",
            "수량": str(n.quantity or 0),
            "고효율 인증 여부": cert_val,
            "공고일자": _as_date(n.notice_date).isoformat() if n.notice_date else "",
            "DETAIL_LINK": n.detail_link or "",
            "KAPT_CODE": n.kapt_code or "",
            "IS_NEW": is_new,
        })

    df = pd.DataFrame(data)
    session.close()
    return df, total_items


def search_data():
    # 안전한 엔진 체크
    if 'engine' in globals() and engine is not None:
        try:
            insp = inspect(engine)
            if not insp.has_table("notices"):
                Base.metadata.create_all(engine)
        except Exception:
            pass

    st.session_state["page"] = 1

    try:
        df, total_items = load_data_from_db(
            st.session_state["office"], st.session_state["source"],
            st.session_state["start_date"], st.session_state["end_date"],
            st.session_state["keyword"], st.session_state["only_cert"],
            st.session_state["include_unknown"], st.session_state["page"],
        )
        st.session_state.df_data = df
        st.session_state.total_items = total_items
    except Exception as e:
        st.error(f"데이터 조회 중 오류가 발생했습니다: {e}")
        st.session_state.df_data = pd.DataFrame()
        st.session_state.total_items = 0

    st.session_state.total_pages = (
        max(1, math.ceil(st.session_state.total_items / ITEMS_PER_PAGE))
        if st.session_state.total_items > 0
        else 1
    )
    st.session_state["data_initialized"] = True




# =========================================================
# 5) 자동 업데이트 스케줄러 (유지)
# =========================================================
import os, threading
from datetime import datetime
import time

from collect_data import run_all_collections  # ✅ 함수명 교체

def run_collection_job():
    """자동수집 스케줄러가 호출하는 래퍼 함수"""
    try:
        logger.info("[Auto-Sync] Starting collection job...")
        run_all_collections()  # ✅ collect_all → run_all_collections 변경
        logger.info("[Auto-Sync] Completed successfully.")
    except Exception as e:
        logger.exception("[Auto-Sync Error] %s", e)


def start_auto_update_scheduler():
    """자동 업데이트 스케줄러 (단일 실행 가드 포함)"""
    if os.getenv("RUN_SCHEDULER", "0") != "1":
        print("스케줄러 실행 스킵 (RUN_SCHEDULER != 1)")
        return

    def scheduler_loop():
        last_run_hour = -1
        while True:
            now = datetime.now()
            if now.hour in [8, 12, 19]:
                if now.minute == 0 and now.hour != last_run_hour:
                    print(f"[Auto-Sync] {now}")
                    try:
                        # 기존 자동 수집 함수 호출
                        run_collection_job()
                    except Exception as e:
                        print(f"[Auto-Sync Error] {e}")
                    last_run_hour = now.hour
            time.sleep(60)

    threading.Thread(target=scheduler_loop, daemon=True).start()
    print(">>> 자동 업데이트 스케줄러 스레드가 시작되었습니다.")


# =========================================================
# 3. 상세 보기 / 즐겨찾기 (수정)
# =========================================================




def _ensure_phone_inline(notice_id: int):
    session = get_db_session()
    if not session: return
    n = session.query(Notice).filter(Notice.id == notice_id).first()

    if (n.source_system or "").upper() != "K-APT" or (n.phone_number or "").strip():
        session.close()
        return

    code = (n.kapt_code or "").strip()
    if not code:
        session.close()
        return

    try:
        basic = fetch_kapt_basic_info(code) or {}
        tel_raw = (basic.get("kaptTel") or "").strip()
        if not tel_raw:
            session.close()
            return

        tel_digits = only_digits_gui(tel_raw)
        n.phone_number = tel_digits
        session.add(n)
        session.commit()

        load_data_from_db.clear()
        _get_new_item_counts_by_source_and_office.clear()
    except Exception as e:
        session.rollback()
        print(f"전화번호 보정 실패: {e}")
    finally:
        session.close()

# =========================================================
# 6. 상세 보기 패널
# =========================================================

def _show_kapt_detail_panel(rec: dict):
    # ✅ 다양한 형태의 단지 코드 필드명을 모두 대응
    kapt_code = (
        rec.get("KAPT_CODE")
        or rec.get("APT_CODE")
        or rec.get("kapt_code")
        or rec.get("apt_code")
    )
    if not kapt_code:
        st.error("단지 코드가 없어 상세 정보를 조회할 수 없습니다.")
        # 기본정보라도 표시
        st.write(f"**사업명:** {rec.get('사업명', '-')}")
        st.write(f"**기관명:** {rec.get('기관명', '-')}")
        st.write(f"**공고일자:** {rec.get('공고일자', '-')}")
        return

    _ensure_phone_inline(rec["id"])

    with st.spinner("단지 정보를 불러오는 중..."):
        basic_info = fetch_kapt_basic_info(kapt_code) or {}
        maint_history = fetch_kapt_maintenance_history(kapt_code) or []

    st.markdown("###### 기본정보")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.text(f"공고명: {rec.get('사업명', '')}")
            st.text(f"도로명주소: {basic_info.get('doroJuso', '정보 없음')}")
            st.text(f"총 동수: {_fmt_int_commas(basic_info.get('kaptDongCnt'))}")
            st.text(f"난방방식: {basic_info.get('codeHeatNm', '정보 없음')}")
        with c2:
            st.text(f"단지명: {basic_info.get('kaptName', '정보 없음')}")
            st.text(f"총 세대수: {_fmt_int_commas(basic_info.get('kaptdaCnt'))}")
            st.text(f"준공일: {_fmt_date_hyphen(basic_info.get('kaptUsedate'))}")
            st.text(f"주택관리방식: {basic_info.get('codeMgrNm', '정보 없음')}")

    st.markdown("###### 관리사무소 정보")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.text(f"관리사무소 연락처: {_fmt_phone_hyphen(basic_info.get('kaptTel'))}")
        with c2:
            st.text(f"관리사무소 팩스: {_fmt_phone_hyphen(basic_info.get('kaptFax'))}")

    st.markdown("###### 유지관리 이력")
    with st.container(border=True):
        if maint_history:
            if isinstance(maint_history, dict): maint_history = [maint_history]
            df_hist = pd.DataFrame(maint_history)
            col_map = {
                "parentParentName": "구분", "parentName": "공사 종별",
                "mnthEtime": "최근 완료일", "year": "수선주기(년)", "useYear": "경과년수"
            }
            existing_cols = [k for k in col_map.keys() if k in df_hist.columns]
            df_display = df_hist[existing_cols].rename(columns=col_map)
            df_display.index = df_display.index + 1

            def highlight_expired(row):
                styles = [''] * len(row)
                try:
                    p_str = str(row.get("수선주기(년)", "0"))
                    e_str = str(row.get("경과년수", "0"))
                    p = int(float(p_str)) if p_str.replace('.', '', 1).isdigit() else 0
                    e = int(float(e_str)) if e_str.replace('.', '', 1).isdigit() else 0
                    
                    if p > 0 and e >= p:
                        return ['background-color: #FFF0F0; color: #D00000; font-weight: bold'] * len(row)
                except: pass
                return styles

            st.dataframe(
                df_display.style.apply(highlight_expired, axis=1),
                use_container_width=True, height=300
            )
        else:
            st.info("유지관리 이력이 없습니다.")

    st.markdown("---")
    st.caption("💡 검색팁: 공고명 또는 단지명을 복사하여, 공동주택 입찰(K-APT) 사이트에서 검색하세요")

    col1, col2, col3 = st.columns([1, 1, 1.5])
    with col1:
        st.code(rec.get('사업명', ''), language=None)
        st.caption("▲ 공고명")
    with col2:
        st.code(basic_info.get('kaptName', ''), language=None)
        st.caption("▲ 단지명")
    with col3:
        st.write("")
        st.link_button("🌐 공동주택 입찰(K-APT) 열기", "https://www.k-apt.go.kr/bid/bidList.do", use_container_width=True)


def _show_dlvr_detail_panel(rec: dict):
    link = rec.get("DETAIL_LINK", "")
    try:
        req_no = link.split(":", 1)[1].split("|", 1)[0].split("?", 1)[0].strip()
    except:
        st.error("납품요구번호 파싱 실패")
        return

    with st.spinner("상세 정보를 불러오는 중..."):
        header = fetch_dlvr_header(req_no) or {}
        items = fetch_dlvr_detail(req_no) or []

    dlvr_req_dt = _pick(header, "dlvrReqRcptDate", "rcptDate")
    req_name    = _pick(header, "dlvrReqNm", "reqstNm", "ttl") or rec.get('사업명', '')
    total_amt_api = _pick(header, "dlvrReqAmt", "totAmt")
    dminst_nm   = _pick(header, "dminsttNm", "dmndInsttNm") or rec.get('기관명', '')
    
    calc_amt = sum([float(i.get("prdctAmt") or 0) for i in items]) if items else 0
    final_amt_str = _fmt_int_commas(total_amt_api if total_amt_api else calc_amt)

    st.markdown("###### 기본정보")
    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.text(f"납품요구번호: {req_no}")
            st.text(f"요청명: {req_name}")
            st.text(f"기관명: {dminst_nm}")
        with c2:
            st.text(f"납품요구일자: {_fmt_date_hyphen(dlvr_req_dt)}")
            st.text(f"납품금액: {final_amt_str}")

    st.markdown("###### 요청물품목록 (행을 클릭하여 선택)")
    
    selected_id = ""
    selected_model = ""
    
    with st.container(border=True):
        if items:
            df_rows = []
            for idx, it in enumerate(items):
                raw_name = _pick(it, "prdctIdntNoNm", "prdctNm", "itemNm")
                nm, model, spec = _split_prdct_name(raw_name)
                amt_val = float(_pick(it, "prdctAmt", "amt", default="0"))
                
                df_rows.append({
                    "순번": idx + 1,
                    "물품분류번호": _pick(it, "prdctClsfNo", "goodClsfNo", "itemClassNo"),
                    "물품식별번호": _pick(it, "prdctIdntNo", "itemNo"),
                    "품명": nm,
                    "모델": model,
                    "규격": spec,
                    "단위": _pick(it, "unitNm", "unit"),
                    "수량": _fmt_int_commas(_pick(it, "prdctQty", "qty", default="0")),
                    "금액(원)": _fmt_int_commas(amt_val)
                })
            
            df = pd.DataFrame(df_rows)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(resizable=True, sortable=True, minWidth=80)
            
            # ✅ id 컬럼 숨기기
            #if "id" in df.columns:
            #    gb.configure_column("id", hide=True)



            gb.configure_selection(
                selection_mode="single", use_checkbox=False, pre_selected_rows=[0]
            )
            
            gb.configure_column("순번", width=60, cellStyle={'textAlign': 'center'})
            gb.configure_column("품명", width=200)
            
            grid_options = gb.build()

            grid_response = AgGrid(
                df, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=250, theme="alpine", allow_unsafe_jscode=True, key=f"dlvr_grid_{req_no}"
            )

            selected_rows = grid_response.get("selected_rows", None)
            row = None

            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                row = selected_rows.iloc[0]
            elif isinstance(selected_rows, list) and len(selected_rows) > 0:
                row = selected_rows[0]
            if row is None and not df.empty:
                row = df.iloc[0]

            if row is not None:
                try:
                    selected_id = row.get("물품식별번호")
                    selected_model = row.get("모델")
                except AttributeError: 
                    selected_id = row["물품식별번호"]
                    selected_model = row["모델"]
            else:
                st.warning("선택된 물품 내역 또는 기본 데이터를 찾을 수 없습니다.")
                selected_id = None
                selected_model = None

        else:
            st.info("물품 내역이 없습니다.")

    st.markdown("---")
    st.caption(f"검색 팁: 선택한 **{selected_model or '모델'}** 정보를 아래에서 복사하여 활용하세요.")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**사업명**")
        st.code(req_name, language=None)
        st.link_button("나라장터 열기", "https://www.g2b.go.kr/", use_container_width=True)
        
    with c2:
        st.markdown(f"**물품식별번호**")
        st.code(selected_id, language=None)
        st.link_button("종합쇼핑몰 열기", "https://shop.g2b.go.kr/", use_container_width=True)

    with c3:
        st.markdown(f"**모델명**")
        st.code(selected_model, language=None)
        st.link_button("에너지공단 기기 검색", "https://eep.energy.or.kr/higheff/hieff_intro.aspx", use_container_width=True)

def show_detail_panel(rec: dict):
    if not rec:
        st.info("좌측 목록에서 공고를 선택해주세요.")
        return

    with st.container():
        source = rec.get("구분", "") or rec.get("source_system", "")
        link = rec.get("DETAIL_LINK", "")

        if source == "K-APT":
            _show_kapt_detail_panel(rec)
        elif link.startswith("dlvrreq:"):
            _show_dlvr_detail_panel(rec)
        else:
            st.markdown("###### 공고 상세 정보")
            with st.container(border=True):
                st.text(f"사업명: {rec.get('사업명', '')}")
                st.text(f"기관명: {rec.get('기관명', '')}")
                st.text(f"공고일: {rec.get('공고일자', '')}")
                st.text(f"사업소: {rec.get('사업소', '')}")
                st.text(f"소재지: {rec.get('소재지', '')}")
                st.text(f"연락처: {rec.get('연락처', '')}")
            
            st.markdown("---")
            if link.startswith("http"):
                st.link_button("🌐 원본 공고 열기", link, use_container_width=True)
            else:
                st.warning("상세 링크가 없습니다.")


# =========================================================
# 6-1. 팝업(모달) 래퍼 함수
# =========================================================
import streamlit as st

@st.dialog("상세 정보", width="large")
def popup_detail_panel(rec: dict):
    """AgGrid 선택 시 모달로 상세 표시 (중복 방지)"""
    # 이미 다른 모달이 열려 있으면 경고만 표시하고 종료
    if st.session_state.get("_popup_active", False):
        st.warning("다른 상세 창이 열려 있습니다. 먼저 닫아주세요.")
        return

    st.session_state["_popup_active"] = True
    try:
        show_detail_panel(rec)
    finally:
        # 사용자가 모달을 닫으면 다음 런에서 다시 열 수 있도록 해제
        st.session_state["_popup_active"] = False



def render_detail_html(rec: dict) -> str:
    """새 창에 렌더링할 상세 HTML 구성 (기존 코드 유지)"""
    title = rec.get("사업명", "")
    org = rec.get("기관명", "")
    office = rec.get("사업소", "")
    date_txt = rec.get("공고일자", "")
    model = rec.get("모델명", "")
    qty = rec.get("수량", "")
    addr = rec.get("소재지", "")
    phone = rec.get("연락처", "")

    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h2 {{ margin-bottom: 6px; }}
        .item p {{ margin: 4px 0; }}
        .box {{
            border:1px solid #ddd;
            padding:12px;
            border-radius:8px;
            background:#fafafa;
        }}
    </style>
    </head>
    <body>
        <h2>{title}</h2>
        <div class="box">
            <p><b>구분:</b> {rec.get("구분",'')}</p>
            <p><b>공고일자:</b> {date_txt}</p>
            <p><b>기관명:</b> {org}</p>
            <p><b>사업소:</b> {office}</p>
            <p><b>소재지:</b> {addr}</p>
            <p><b>모델명:</b> {model}</p>
            <p><b>수량:</b> {qty}</p>
            <p><b>연락처:</b> {phone}</p>
        </div>
        <hr>
        <p><b>상세 링크:</b></p>
        <p>{rec.get("DETAIL_LINK","")}</p>
    </body>
    </html>
    """
    return html

# =========================================================
# 4. 공고 리스트 UI (카드형 / 목록형) (수정)
# =========================================================


def render_notice_cards(df: pd.DataFrame):
    if df.empty:
        st.warning("조회된 데이터가 없습니다.")
        return

    DEVICE_KEYWORDS = [
        "led", "엘이디", "발광다이오드", "조명", "가로등", "보안등", "터널등", "스마트 led", "스마트led",
        "모터", "전동기", "펌프", "블로워", "팬", "에어드라이어", "pcm",
        "히트펌프", "냉동기", "터보압축기", "김건조기",
        "변압기", "트랜스", "인버터", "인버터 제어형",
        "공기압축기", "사출성형기",
        "승강기", "엘리베이터"
    ]

    IMPROVEMENT_KEYWORDS = [
        "보수", "개선", "성능개선", "효율개선", "개체", "교체",
        "정비", "개량", "리모델링", "개보수", "노후교체", "업그레이드",
    ]

    ENERGY_KEYWORDS = [
        "고효율", "에너지절감", "효율향상", "에너지절약",
        "전력기금", "지원사업", "보조금", "정부지원",
        "효율등급", "에너지이용합리화"
    ]

    PRIORITY_KEYWORDS = DEVICE_KEYWORDS + IMPROVEMENT_KEYWORDS + ENERGY_KEYWORDS

    records = df.to_dict(orient="records")
    per_row = 2

    for i in range(0, len(records), per_row):
        row = records[i:i+per_row]
        cols = st.columns(per_row)

        for col, rec in zip(cols, row):
            with col:
                title = rec.get("사업명", "")
                org = rec.get("기관명", "")
                office = rec.get("사업소", "")
                gubun = rec.get("구분", "")
                date_txt = rec.get("공고일자", "")
                is_new = rec.get("IS_NEW", False)

                badge_new = ('<span style="color:#d84315;font-weight:bold;"> NEW</span>' if is_new else "")

                # 🔍 제목에서 키워드 찾기
                matched_kw = None
                t = title.lower()
                for kw in PRIORITY_KEYWORDS:
                    if kw.lower() in t:
                        matched_kw = kw
                        break

                keyword_badge = ""
                if matched_kw:
                    keyword_badge = (
                        f"<span style='background-color:#e8f0fe;color:#1a73e8;"
                        f"padding:2px 6px;border-radius:10px;font-size:11px;"
                        f"white-space:nowrap; margin-left:6px;'>{matched_kw}</span>"
                    )

                # ⚠ HTML 시작 부분 절대 들여쓰기 하지 말 것!!
                card_html = f"""<div style='border:1px solid #ddd; border-radius:10px; padding:12px 14px;
background:#ffffff; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,0.05); height:170px;'>
<div style="display:flex; justify-content:space-between; align-items:center; font-size:14px; color:#555;">
    <div><b>{gubun}</b> | {date_txt}{badge_new}</div>
    <div>{keyword_badge}</div>
</div>
<div style='font-size:17px; font-weight:600; margin-top:8px; line-height:1.3; word-break:keep-all;'>
    {title}
</div>
<div style='font-size:14px;color:#666;margin-top:8px;'>
    <b>{org}</b> | {office}
</div>
</div>"""

                st.markdown(card_html, unsafe_allow_html=True)

                if st.button("🔍 상세", key=f"detail_card_{rec['id']}", use_container_width=True):
                    popup_detail_panel(rec)


def render_notice_table(df):
    st.markdown("### 📋 공고 목록")

    if df.empty:
        st.info("표시할 공고가 없습니다.")
        return None

    # 원본 데이터 백업
    df_full = df.copy()

    # ✅ 상세 아이콘 추가
    df_disp = df_full.copy()
    df_disp.insert(0, "상세", "🔍")

    # ✅ NEW 표시 로직
    def format_title(row):
        title = row.get("사업명", "")
        prefixes = []
        source = row.get("구분")
        pub_date_str = row.get("공고일자")
        is_existing_new = row.get("IS_NEW")

        is_real_new = False
        try:
            if pub_date_str:
                pub_date_str = str(pub_date_str).replace('.', '-')
                pub_date = pd.to_datetime(pub_date_str, errors='coerce').normalize()
                if not pd.isna(pub_date):
                    today = pd.Timestamp.now().normalize()
                    limit_date = today - BusinessDay(2)
                    if pub_date >= limit_date:
                        is_real_new = True
        except Exception:
            is_real_new = False

        if source == "K-APT" and is_real_new:
            prefixes.append("🔵 [NEW]")
        elif is_existing_new:
            prefixes.append("🔴 [NEW]")

        return f"{' '.join(prefixes)} {title}" if prefixes else title

    df_disp["사업명"] = df_disp.apply(format_title, axis=1)

    # ✅ 표시 컬럼 정의 (id 숨기기, APT_CODE 유지)
    visible_cols = [
        "상세", "순번", "구분", "사업소", "단계", "사업명",
        "기관명", "소재지", "연락처", "모델명", "수량",
        "고효율 인증 여부", "공고일자", "APT_CODE"
    ]
    final_cols = [c for c in visible_cols if c in df_disp.columns]

    # ✅ 원본 인덱스 저장용 숨김 컬럼
    df_disp["__ROW_ID"] = df_disp.index
    df_disp = df_disp[[*final_cols, "__ROW_ID"]]

    from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_column("상세", width=80, pinned="left")
    gb.configure_column("__ROW_ID", hide=True)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_disp,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.FILTERED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=520,
        fit_columns_on_grid_load=True,
        theme="alpine",
        allow_unsafe_jscode=False,
        key="notice_grid_main"
    )

    selected_rows = grid_response.get("selected_rows", [])
    if not selected_rows:
        return None

    # ✅ 원본 레코드 복원 (KAPT_CODE 등 숨은 컬럼 포함)
    try:
        rid = int(selected_rows[0]["__ROW_ID"])
        rec = df_full.loc[rid].to_dict()
    except Exception:
        rec = selected_rows[0]

    # ✅ 중복 호출 방지 및 디바운스
    if (
        not st.session_state.get("_popup_active", False)
        and st.session_state.get("_last_selected_row_id") != rid
    ):
        st.session_state["_last_selected_row_id"] = rid
        popup_detail_panel(rec)

    return rec





# =========================================================
# 5. 메인 페이지 (공고 조회 및 검색) (수정)
# =========================================================

def main_page():
    # 💡 간편 검색 버튼 클릭 처리를 위한 헬퍼 함수
    def set_keyword_and_search(kw):
        st.session_state["keyword"] = kw
        st.session_state["page"] = 1
        search_data()
        st.rerun()

    st.markdown("""
        <style>
        .keyword-btn {
            display: inline-flex; align-items: center; justify-content: center;
            padding: 5px 10px; min-width: 90px; height: 32px; white-space: nowrap;
            border: 1px solid #ccc; border-radius: 6px; margin: 4px;
            background: #f8f8f8; font-size: 13px;
        }
        .keyword-btn:hover { background: #eee; }
        .stButton>button[kind="secondary"] {
            border-color: #ccc;
        }
        </style>
        """, unsafe_allow_html=True
    )



    st.markdown(
        """
        <div style="
            text-align:center;
            background:linear-gradient(135deg, #f3f7ff, #e9eef9);
            padding: 1.8rem 0 1.6rem 0;
            border-radius: 14px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.07);
            margin-bottom: 1.8rem;
            font-family: 'Pretendard', 'Segoe UI', sans-serif;
        ">
            <h1 style="
                font-weight:650;
                color:#003EAA;
                letter-spacing:-0.5px;
                margin-bottom:0.4rem;
                font-size:1.6rem; 
            ">
                EERS 업무 지원 시스템
            </h1>
            <p style="
                font-size:1.08rem;
                color:#444;
                margin-top:0;
                margin-bottom:0.3rem;
            ">
                나라장터·K-APT <strong>입찰정보를 간편하게 조회</strong>하고,<br>
                고효율기기 <strong>수요 현황을 한눈에 확인</strong>하세요.
            </p>
            <p style="
                font-size:0.95rem;
                color:#666;
                margin-top:0.8rem;
            ">
                대구본부 에너지효율부 EERS팀
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



    st.subheader("🔍 검색 조건")

    # 💡 검색 조건 변경 시 즉시 검색
    col1, col2, col3 = st.columns([1.5, 1.5, 4])
    new_counts = _get_new_item_counts_by_source_and_office()
    current_office = st.session_state.get("office", "전체")
    office_counts = new_counts.get(current_office, {"G2B": 0, "K-APT": 0})

    # -------------------------
    # 좌측: 사업소 / 데이터 출처
    # -------------------------
    with col1:
        st.selectbox("사업소 선택", options=OFFICES, key="office", on_change=search_data)
        st.selectbox("데이터 출처", options=["전체", "나라장터", "K-APT"], key="source", on_change=search_data)

    # -------------------------
    # 중앙: 날짜
    # -------------------------
    with col2:
        st.date_input("시작일", key="start_date", min_value=MIN_SYNC_DATE, on_change=search_data)
        st.date_input("종료일", key="end_date", max_value=DEFAULT_END_DATE, on_change=search_data)

    # -------------------------
    # 우측: 키워드 검색 + 검색 버튼
    # -------------------------
    with col3:

        col3_1, col3_2 = st.columns([4, 1])

        with col3_1:
            # keyword_override 적용
            if "keyword_override" in st.session_state:
                default_kw = st.session_state["keyword_override"]
                del st.session_state["keyword_override"]
            else:
                default_kw = st.session_state.get("keyword", "")

            st.text_input(
                "키워드 검색",
                placeholder="예: led, 변압기...",
                key="keyword",
                value=default_kw
            )

        with col3_2:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            st.button("검색", on_click=search_data, type="primary", use_container_width=True)

        # 체크박스 영역
        col3_checkbox_1, col3_checkbox_2, _ = st.columns([1, 1, 3])
        with col3_checkbox_1:
            st.checkbox("고효율(인증)만 보기", key="only_cert", on_change=search_data)
        with col3_checkbox_2:
            st.checkbox("관할불명 포함", key="include_unknown", on_change=search_data)

    

    # --------------------------------
    # 데이터 로딩
    # --------------------------------
    if not st.session_state.get("data_initialized", False):
        search_data()
        st.session_state["data_initialized"] = True

    df = st.session_state.df_data

    if df.empty:
        st.warning("조회된 데이터가 없습니다.")
        return

    df = df.reset_index(drop=True)
    df["순번"] = df.index + 1

    # --------------------------------
    # 카드형 / 목록형 UI 선택
    # --------------------------------
    view_col1, _ = st.columns([1, 6])
    with view_col1:
        view_choice = st.radio(
            "보기 방식",
            ["카드형", "목록형"],
            horizontal=True,
            key="view_mode_radio",
            index=["카드형", "목록형"].index(st.session_state.get("view_mode", "카드형"))
        )
        st.session_state["view_mode"] = view_choice

    selected_rec = None
    if st.session_state["view_mode"] == "카드형":
        render_notice_cards(df)
    else:
        st.caption("💡 돋보기 아이콘을 클릭하면 상세 팝업이 열립니다.")
        selected_rec = render_notice_table(df)

    if selected_rec:
        popup_detail_panel(selected_rec)

    # 페이징 생략

def calc_progress(df):
    """'신규' 또는 '갱신' 항목만 진행률에 포함"""
    filtered = df[df["process_state"].isin(["NEW", "UPDATED"])]
    if len(df) == 0:
        return 0
    return round(len(filtered) / len(df) * 100, 2)

def data_sync_page():
    st.title("🔄 데이터 업데이트")

    if not has_sync_access():
        st.error("데이터 수집 권한이 없습니다.")
        return

    # --- 마지막 실행 시각 표시 ---
    last_dt = _get_last_sync_datetime_from_meta()
    last_txt = last_dt.strftime("%Y-%m-%d %H:%M") if last_dt else "기록 없음"
    st.info(f"마지막 API 호출 일시: **{last_txt}**")
    st.markdown("---")

    # --- 날짜 설정 UI ---
    st.subheader("기간 설정")
    col_preset1, col_preset2 = st.columns(2)

    def set_sync_today():
        st.session_state["sync_start"] = date.today()
        st.session_state["sync_end"] = date.today()

    def set_sync_week():
        today = date.today()
        start = today - timedelta(days=6)
        st.session_state["sync_start"] = max(start, MIN_SYNC_DATE)
        st.session_state["sync_end"] = today

    if col_preset1.button("오늘 하루만 업데이트"):
        set_sync_today()
        st.rerun()

    if col_preset2.button("최신 1주일 업데이트"):
        set_sync_week()
        st.rerun()

    col_date1, col_date2 = st.columns([1, 1])
    if "sync_start" not in st.session_state or "sync_end" not in st.session_state:
        set_sync_today()

    with col_date1:
        start_date = st.date_input("시작일", min_value=MIN_SYNC_DATE, key="sync_start")
    with col_date2:
        end_date = st.date_input("종료일", max_value=DEFAULT_END_DATE, key="sync_end")

    st.caption("권장: 하루 단위로 업데이트하거나, 최근 1주/1개월 단위로 진행해 주세요. (API 한도 유의)")
    st.markdown("---")

    # --- 동기화 실행 ---
    if st.button("선택 기간 업데이트 시작", type="primary", key="start_sync_btn"):
        if start_date > end_date:
            st.error("시작일은 종료일보다 늦을 수 없습니다.")
            st.stop()
        if (end_date - start_date).days >= 92:
            st.error("조회 기간은 최대 92일(3개월)까지만 가능합니다.")
            st.stop()

        st.session_state["is_updating"] = True

        # === 진행 상태 표시 영역 ===
        st.subheader("📊 데이터 수집 진행률")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_placeholder = st.empty()

        # === 초기 변수 ===
        dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        stages_to_run = list(STAGES_CONFIG.values())
        total_steps = len(dates) * len(stages_to_run)
        current_step = 0

        # === 로그/중복 관리 ===
        sync_logs = []
        st.session_state.setdefault("_printed_done_msgs", set())
        st.session_state.setdefault("_last_log_line", "")

        # --- 로그 함수 ---
        def append_log(msg: str):
            """중복 방지 + 실시간 UI 반영"""
            if st.session_state["_last_log_line"] == msg:
                return
            if msg.startswith("✔") and msg in st.session_state["_printed_done_msgs"]:
                return

            sync_logs.append(msg)
            st.session_state["_last_log_line"] = msg
            if msg.startswith("✔"):
                st.session_state["_printed_done_msgs"].add(msg)

            # ✅ 로그 덮어쓰기 (누적 X)
            log_placeholder.code("\n".join(sync_logs[-200:]), language="text")

        # --- 실행부 ---
        try:
            for d in dates:
                disp_date = d.strftime("%Y-%m-%d")

                for stage in stages_to_run:
                    name = stage.get("name", "Unknown Stage")

                    # ✅ 현재 단계 표시 (덮어쓰기)
                    status_text.markdown(f"**현재:** `{disp_date}` · **{name}**")
                    append_log(f"▶ [{disp_date}] {name} 수집 시작")

                    try:
                        # 실제 수집 실행
                        fetch_data_for_stage(d.strftime("%Y%m%d"), stage)
                        append_log(f"✔ [{disp_date}] {name} 완료")
                    except Exception as e:
                        append_log(f"❌ [{disp_date}] {name} 오류: {e}")
                        logger.error(f"[SYNC] {disp_date} {name} 오류: {e}", exc_info=True)

                    # ✅ 진행률 갱신 (덮어쓰기)
                    current_step += 1
                    pct = int(current_step / total_steps * 100)
                    progress_bar.progress(pct / 100)
                    status_text.markdown(f"**진행률:** {pct}% ({current_step}/{total_steps})")

            # --- 완료 처리 ---
            progress_bar.progress(1.0)
            status_text.success("🎉 전체 작업 완료!")
            append_log("✅ 모든 단계가 정상 완료되었습니다.")

            # 캐시 초기화 및 메타데이터 업데이트
            _set_last_sync_datetime_to_meta(datetime.now())
            load_data_from_db.clear()
            _get_new_item_counts_by_source_and_office.clear()

            st.success("데이터 수집이 완료되었습니다. 상단 '공고 조회 및 검색'에서 다시 조회해 주세요.")
            st.session_state["is_updating"] = False
            st.rerun()

        except Exception as global_e:
            status_text.error(f"⚠️ 동기화 작업 중 오류 발생: {global_e}")
            logger.error(f"Global Sync Error: {global_e}", exc_info=True)

        finally:
            st.session_state["is_updating"] = False



def data_status_page():
    st.title("📅 데이터 현황 보기")

    col_office, _ = st.columns([1, 2])
    with col_office:
        selected_office = st.selectbox("사업소 필터", OFFICES, key="status_office_select")

    @st.cache_data(ttl=300)
    def get_all_db_notice_dates(target_office):
        session = get_db_session()
        if not session: return set()
        try:
            query = session.query(Notice.notice_date)
            
            if target_office and target_office != "전체":
                query = query.filter(
                    or_(
                        Notice.assigned_office == target_office,
                        Notice.assigned_office.like(f"{target_office}/%"),
                        Notice.assigned_office.like(f"%/{target_office}"),
                        Notice.assigned_office.like(f"%/{target_office}/%"),
                    )
                )
                
            dates_raw = query.distinct().all()
            dates = [_as_date(d[0]) for d in dates_raw]
            
            today = date.today()
            return {d for d in dates if d and d <= today}
        except Exception:
            return set()
        finally:
            session.close()

    data_days_set = get_all_db_notice_dates(selected_office)

    today = date.today()
    
    if "status_year" not in st.session_state: st.session_state["status_year"] = today.year
    if "status_month" not in st.session_state: st.session_state["status_month"] = today.month

    col_year, col_month = st.columns(2)
    with col_year:
        year = st.number_input("연도", min_value=2020, max_value=2030, 
                               value=st.session_state["status_year"], key="status_year_input")
    with col_month:
        month = st.number_input("월", min_value=1, max_value=12, 
                                value=st.session_state["status_month"], key="status_month_input")

    st.session_state["status_year"] = year
    st.session_state["status_month"] = month

    st.markdown("---")
    st.markdown(f"### 🗓️ {year}년 {month}월 ({selected_office})")

    cal = calendar.Calendar()
    month_days = cal.monthdayscalendar(year, month)

    cols = st.columns(7)
    weekdays = ["일", "월", "화", "수", "목", "금", "토"]
    for i, w in enumerate(weekdays):
        cols[i].markdown(f"<div style='text-align:center; font-weight:bold;'>{w}</div>", unsafe_allow_html=True)

    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].write("")
            else:
                current_date = date(year, month, day)
                has_data = current_date in data_days_set
                
                btn_type = "primary" if has_data else "secondary"
                label = f"{day}"
                
                btn_key = f"cal_btn_{selected_office}_{year}_{month}_{day}"
                
                if cols[i].button(label, key=btn_key, type=btn_type, use_container_width=True):
                    if has_data:
                        st.session_state["status_selected_date"] = current_date
                    else:
                        st.toast(f"{month}월 {day}일에는 '{selected_office}' 관련 데이터가 없습니다.")

    if "status_selected_date" in st.session_state:
        sel_date = st.session_state["status_selected_date"]
        
        if sel_date.year == year and sel_date.month == month:
            st.markdown("---")
            st.markdown(f"### 📂 {sel_date.strftime('%Y-%m-%d')} 데이터 목록")
            
            session = get_db_session()
            if not session:
                st.error("DB 연결 오류")
                return
            date_str = sel_date.isoformat()
            
            query = session.query(Notice).filter(Notice.notice_date == date_str)
            
            if selected_office != "전체":
                query = query.filter(
                    or_(
                        Notice.assigned_office == selected_office,
                        Notice.assigned_office.like(f"{selected_office}/%"),
                        Notice.assigned_office.like(f"%/{selected_office}"),
                        Notice.assigned_office.like(f"%/{selected_office}/%"),
                    )
                )
            
            rows = query.order_by(Notice.id.desc()).all()
            session.close()

            if rows:
                data = []
                for n in rows:
                        data.append({
                            "id": n.id,
                            "구분": "K-APT" if n.source_system == "K-APT" else "나라장터",
                            "사업소": (n.assigned_office or "").replace("/", " "),
                            "단계": n.stage or "",
                            "사업명": n.project_name or "",
                            "기관명": n.client or "",
                            "소재지": n.address or "",
                            "연락처": fmt_phone(n.phone_number or ""),
                            "모델명": n.model_name or "",
                            "수량": str(n.quantity or 0),
                            "고효율 인증 여부": _normalize_cert(n.is_certified),
                            "공고일자": date_str,
                            "DETAIL_LINK": n.detail_link or "",
                            "KAPT_CODE": n.kapt_code or "",
                            "IS_NEW": False
                        })

                
                df_day = pd.DataFrame(data)
                
                rec = render_notice_table(df_day)
                
                if rec: popup_detail_panel(rec)
            else:
                st.info("해당 조건의 데이터가 없습니다.")




# === Dialog & Selection Guard (once) ===
import streamlit as st

if "_popup_active" not in st.session_state:
    st.session_state["_popup_active"] = False

if "_last_selected_row_id" not in st.session_state:
    st.session_state["_last_selected_row_id"] = None




# =========================================================
# 7. 관리자 인증 / 사이드바 / 전체 앱 실행 (최종 수정)
# =========================================================


def eers_app():
    import streamlit as st

    st.markdown(
        """
        <link rel="manifest" href="manifest.json">
        <link rel="icon" type="image/png" sizes="192x192" href="eers_icon_192.png">
        <link rel="apple-touch-icon" href="eers_icon_512.png">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="theme-color" content="#0046AD">
        """,
        unsafe_allow_html=True
    )



    if "cookie_manager_instance" not in st.session_state:
        st.session_state["cookie_manager_instance"] = stx.CookieManager(key="eers_cookie_manager")

    init_session_state()
    
    # [쿠키 기반 로그인 상태 복구]
    cookie_manager = st.session_state["cookie_manager_instance"]


    # [사이드바 구성]
    with st.sidebar:
        st.header("EERS 업무 지원 시스템")
        



        
        st.markdown("---")

        # ---------------------------
        # 메뉴 렌더링 함수
        # ---------------------------
        def render_menu_button(name):
            current = st.session_state.get("route_page", "공고 조회 및 검색")
            btn_type = "primary" if current == name else "secondary"
            if st.button(name, use_container_width=True, type=btn_type, key=f"menu_{name}"):
                st.session_state["route_page"] = name
                st.rerun()


        # ---------------------------
        # 메뉴 영역 구성
        # ---------------------------
        st.markdown("### 📌 메인 기능")
        render_menu_button("공고 조회 및 검색")
        render_menu_button("데이터 현황")

        # ✅ 관리자 전용 메뉴
        if has_sync_access():
            st.markdown("---")
            st.caption("🔒 관리자 전용")
            render_menu_button("데이터 업데이트")


        st.markdown("---")
        
        st.subheader("관련 사이트")

        def open_new_tab(url):
            st.components.v1.html(f"<script>window.open('{url}', '_blank');</script>", height=0, width=0)
        
        if st.button("나라장터", key="link_g2b", use_container_width=True): open_new_tab("https://www.g2b.go.kr/")
        if st.button("에너지공단", key="link_energy", use_container_width=True): open_new_tab("https://eep.energy.or.kr/higheff/hieff_intro.aspx")
        if st.button("K-APT", key="link_kapt", use_container_width=True): open_new_tab("https://www.k-apt.go.kr/bid/bidList.do")
        if st.button("한전ON", key="link_kepco", use_container_width=True): open_new_tab("https://home.kepco.co.kr/kepco/CY/K/F/CYKFPP001/main.do?menuCd=FN0207")
        if st.button("에너지마켓 신청", key="link_enmarket", use_container_width=True): open_new_tab("https://en-ter.co.kr/ft/biz/eers/eersApply/info.do")

        # ==========================
        # 사이드바 맨 아래 - 데이터 수집 캡션
        # ==========================


        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

        render_sidebar_sync_caption()

    # [페이지 라우팅]
    page = st.session_state.route_page
    if page == "공고 조회 및 검색":
        main_page()
    elif page == "데이터 현황":
        data_status_page()
    elif page == "데이터 업데이트":
        data_sync_page()
    else:
        main_page()



if __name__ == "__main__":
    if engine and not inspect(engine).has_table("notices"):
        Base.metadata.create_all(engine)
    # app 시작 시 한 번만
    start_auto_update_scheduler()

    eers_app()
