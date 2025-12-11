import streamlit as st
import re
import pandas as pd
import math
import sys
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Dict
from sqlalchemy import or_, func, inspect
from sqlalchemy import event
import calendar
from io import BytesIO
import html
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
import threading
import time
import smtplib
from email.message import EmailMessage
import random
import string
import extra_streamlit_components as stx
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import ssl
import threading
from datetime import datetime
import time
# 표준 Python 로깅 모듈 사용
import logging 
# 로거 설정 (Streamlit Cloud의 Console Logs에 기록됩니다)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================================
# 0. config/Secrets 안전 로딩 (Cloud 대응)
# =======================================
try:
    import config as _local_config
except ModuleNotFoundError:
    _local_config = None

def _cfg(name, default=None):
    # 1) 로컬 config.py에 있으면 그 값 사용
    if _local_config is not None and hasattr(_local_config, name):
        return getattr(_local_config, name)

    # 2) 없으면 Streamlit Cloud secrets에서 읽기
    try:
        return st.secrets[name]
    except Exception:
        return default
    
DATABASE_URL = _cfg("DATABASE_URL", "")    


# 1) 환경변수(Fly.io, Docker)를 최우선
env_db_url = os.environ.get("SUPABASE_DATABASE_URL")

# 2) Streamlit secrets 또는 config.py 값
cfg_db_url = _cfg("SUPABASE_DATABASE_URL", "")

SUPABASE_DATABASE_URL = env_db_url or cfg_db_url

if not SUPABASE_DATABASE_URL:
    st.error("FATAL: SUPABASE_DATABASE_URL이 환경변수 또는 secrets/config에 없습니다.")
    st.stop()  # 앱 중단 (필수)

    
# 메일 관련 설정
MAIL_FROM       = _cfg("MAIL_FROM", "")
MAIL_SMTP_HOST  = _cfg("MAIL_SMTP_HOST", "")
MAIL_SMTP_PORT  = int(_cfg("MAIL_SMTP_PORT", 587) or 587)
MAIL_USER       = _cfg("MAIL_USER", "")
MAIL_PASS       = _cfg("MAIL_PASS", "")

# 관리자 비밀번호
ADMIN_PASSWORD  = _cfg("ADMIN_PASSWORD", "admin")

# 메일 발신자 이름
MAIL_FROM_NAME  = _cfg("MAIL_FROM_NAME", "대구본부 EERS팀")

# 최소 동기화 시작일
from datetime import date as _date_cls
_min_sync_raw = _cfg("MIN_SYNC_DATE", _date_cls(2025, 12, 1))
if isinstance(_min_sync_raw, str):
    MIN_SYNC_DATE = _date_cls.fromisoformat(_min_sync_raw)
else:
    MIN_SYNC_DATE = _min_sync_raw

SIX_MONTHS = timedelta(days=30 * 6)

# =========================================================
# 0. 로컬 모듈 및 설정 로드 (PyQt 잔재 및 gui_app 제거)
# =========================================================
# 🛑 DB 연결에 필요한 모든 변수를 초기화합니다.
engine = None
SessionLocal = None
# Note, MailRecipient, MailHistory 클래스는 아래 try 블록에서 가져오거나 더미로 대체됩니다.

try:
    # database.py에서 필요한 모듈과 함수를 임포트합니다.
    from database import (
        get_engine_and_session, # 👈 추가된 함수
        Notice, 
        MailRecipient, 
        MailHistory, 
        Base, 
        engine as db_module_engine, # database.py의 초기 None 엔진
        SessionLocal as db_module_session_local # database.py의 초기 None 세션
    )
    # collect_data, mailer 임포트는 유지합니다.
    from collect_data import (
        fetch_data_for_stage, STAGES_CONFIG, is_relevant_text,
        resolve_address_from_bjd, fetch_kapt_basic_info, fetch_kapt_maintenance_history,
        _as_text, _to_int as _to_int_collect, _extract_school_name, _assign_office_by_school_name
    )
    from mailer import send_mail, build_subject, build_body_html, build_attachment_html

    if SUPABASE_DATABASE_URL:
        logger.info("Connecting to Supabase PostgreSQL (cached)...")

        # ✅ 캐시된 엔진/세션팩토리 사용 (Streamlit rerun 시에도 재사용)
        @st.cache_resource
        def get_engine_cached():
            from database import get_engine_and_session
            return get_engine_and_session(SUPABASE_DATABASE_URL)

        _engine, _SessionLocal = get_engine_cached()

        # 전역 변수로 바인딩
        engine = _engine
        SessionLocal = _SessionLocal

        logger.info("Database connection successful and metadata loaded (cached).")

    else:
        logger.warning("SUPABASE_DATABASE_URL not found. Running with dummy database logic.")


except ImportError as e:
    # 필수 모듈 로드 실패 시, Streamlit이 실행되도록 더미 정의를 유지합니다.
    st.warning(f"경고: 필수 모듈 (database, collect_data, mailer) 로드 실패: {e}. 더미 함수로 대체됩니다.")
    
    # 🛑 Notice, MailRecipient, MailHistory 클래스를 이 블록 내에서 정의해야 합니다.
    class Notice: pass
    class MailRecipient: pass
    class MailHistory: pass
    
    # 더미 객체 정의
    engine = None
    class Base:
        @staticmethod
        def metadata():
            class Meta:
                @staticmethod
                def create_all(eng): pass
            return Meta()
    def SessionLocal(): return None
    
    # ... (collect_data, mailer 더미 함수 정의는 기존과 동일하게 유지)
    def fetch_data_for_stage(*args): pass
    STAGES_CONFIG = {"G2B": {"name": "G2B", "code": "g2b"}, "KAPT": {"name": "K-APT", "code": "kapt"}}
    def fetch_kapt_basic_info(code): return {}
    def fetch_kapt_maintenance_history(code): return []
    def fetch_dlvr_header(req_no): return {}
    def fetch_dlvr_detail(req_no): return []
    def send_mail(**kwargs): return True
    def build_subject(*args): return "테스트 제목"
    def build_body_html(*args): return "<html><body>테스트 본문</body></html>", "첨부.html", "첨부 내용", "미리보기"


# =========================================================
# 0-A. 대체 유틸리티
# =========================================================
def _get_last_sync_datetime_from_meta():
    return datetime.now() - timedelta(hours=2)
def _set_last_sync_datetime_to_meta(dt: datetime):
    pass
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
    return re.sub(r'\D', '', str(val or ''))
def fmt_phone(val):
    v = only_digits_gui(val)
    if not v:
        return "정보 없음"
    if len(v) == 8: return f"{v[:4]}-{v[4:]}"
    if len(v) == 9: return f"{v[:2]}-{v[2:5]}-{v[5:]}"
    if len(v) == 10: return f"{v[:2]}-{v[2:6]}-{v[6:]}" if v.startswith("02") else f"{v[:3]}-{v[3:6]}-{v[6:]}"
    if len(v) == 11: return f"{v[:3]}-{v[3:7]}-{v[7:]}"
    return str(val)

# =========================================================
# 0-1. 상수 및 헬퍼
# =========================================================
OFFICES = [
    "전체", "직할", "동대구지사", "경주지사", "남대구지사", "서대구지사",
    "포항지사", "경산지사", "김천지사", "영천지사", "칠곡지사",
    "성주지사", "청도지사", "북포항지사", "고령지사", "영덕지사",
]
ITEMS_PER_PAGE = 100
DEFAULT_START_DATE = MIN_SYNC_DATE
DEFAULT_END_DATE = date.today()
MAIL_EXCLUDE_OFFICES = ["전체"]
CERT_TRUE_VALUES = {"O", "0", "Y", "YES", "1", "TRUE", "인증"}

def open_new_window(url: str):
    js = f"""
    <script>
        window.open("{url}", "_blank");
    </script>
    """
    st.components.v1.html(js, height=0)

def _normalize_cert(val: str) -> str:
    if val is None: return ""
    s = str(val).strip().upper()
    if not s: return ""
    if s in CERT_TRUE_VALUES: return "O"
    if s in {"X", "N", "NO", "미인증"}: return "X"
    return val

def _fmt_int_commas(val):
    try:
        s = str(val or "").replace(",", "").strip()
        if not s or s.lower() == "none": return "정보 없음"
        n = int(float(s))
        return f"{n:,}"
    except Exception:
        return str(val) if val not in (None, "") else "정보 없음"

def _fmt_date_hyphen(val):
    import re
    s = str(val or "").strip()
    if not s: return "정보 없음"
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
    import re
    v = re.sub(r"\D", "", str(val or ""))
    if not v: return "정보 없음"
    if len(v) == 8: return f"{v[:4]}-{v[4:]}"
    if len(v) == 9: return f"{v[:2]}-{v[2:5]}-{v[5:]}"
    if len(v) == 10: return f"{v[:2]}-{v[2:6]}-{v[6:]}" if v.startswith("02") else f"{v[:3]}-{v[3:6]}-{v[6:]}"
    if len(v) == 11: return f"{v[:3]}-{v[3:7]}-{v[7:]}"
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

def open_popup_window(html_content: str):
    encoded = html_content.replace("'", "\\'")
    js = f"""
    <script>
        var popup = window.open("", "_blank", "width=1200,height=900,scrollbars=yes");
        popup.document.write('{encoded}');
        popup.document.close();
    </script>
    """
    st.components.v1.html(js, height=0)

def _to_int_local(val):
    try:
        return int(str(val).replace(",", "").strip() or 0)
    except Exception:
        return 0

# DB PRAGMA 설정 (SQLite) - 실제 DB 모듈이 있다면 활성화
if engine and "sqlite" in str(engine.url): # <--- PostgreSQL을 위해 조건 변경 또는 삭제
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        # ... (PRAGMA 설정 코드 삭제 또는 주석 처리)
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
# =========================================================
# 로그인 & 인증 관련 함수 (수정)
# =========================================================
# =========================================================
# 로그인 & 인증 관련 설정 및 헬퍼 함수
# (기존 코드에서 유지되는 부분 - send_verification_email은 필수)
# =========================================================

# (import stx, random, string, datetime, timedelta, ssl, smtplib, EmailMessage 등은
# 상위 코드에서 이미 처리되었으므로 생략하고, 변경된 함수만 제시합니다.)

def get_manager():
    """CookieManager 인스턴스를 반환합니다."""
    return st.session_state.get("cookie_manager_instance")

def logout():
    """세션을 초기화하고 쿠키를 삭제하여 로그아웃합니다."""
    manager = st.session_state.get("cookie_manager_instance")
    if manager:
        try:
            manager.delete(cookie="eers_auth_token")
        except Exception as e:
            # print(f"로그아웃: 쿠키 삭제 중 오류 발생 (무시): {e}")
            pass

    # 세션 상태 초기화
    keys_to_delete = [k for k in st.session_state.keys() if k not in ["cookie_manager_instance", "auto_view_initialized"]]
    for k in keys_to_delete:
        del st.session_state[k]
        
    st.toast("로그아웃되었습니다.", icon="👋")
    st.rerun()

def send_verification_email(to_email, code):

    # 🔥 콘솔 출력 (디버그용)
    print("\n==============================")
    print("📧 인증코드 이메일 발송")
    print(f"👉 수신자: {to_email}")
    print(f"🔐 인증코드: {code}")
    print("==============================\n")

    msg = EmailMessage()
    
    plain_content = f"""
    [EERS 시스템 로그인 인증]
    
    인증코드: {code}
    
    위 코드를 시스템에 입력하여 로그인을 완료해주세요.
    """
    msg.set_content(plain_content, subtype="plain") 
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #f9f9f9;">
            <h3 style="color: #333;">[EERS 시스템 로그인 인증]</h3>
            <p>귀하의 로그인 인증 코드는 다음과 같습니다:</p>
            <div style="background-color: #ffe4e1; color: #d9534f; padding: 10px; border-radius: 4px; font-size: 18px; font-weight: bold; text-align: center; margin: 15px 0;">
                {code}
            </div>
            <p>위 코드를 시스템에 입력하여 로그인을 완료해주세요.</p>
        </div>
    </body>
    </html>
    """
    msg.add_alternative(html_content, subtype="html")

    msg["Subject"] = "[EERS] 로그인 인증코드 안내"
    msg["From"] = MAIL_FROM
    msg["To"] = to_email

    context = ssl.create_default_context()
    
    try:
        with smtplib.SMTP(MAIL_SMTP_HOST, MAIL_SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(MAIL_USER, MAIL_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"!!! 메일 발송 실패: {e}")
        return False

# =========================================================
# 재구성된 로그인/인증 UI 렌더링 함수 (핵심)
# =========================================================

def check_auth_cookie():
    """쿠키를 확인하고 유효하면 로그인 상태를 설정합니다."""
    if "cookie_manager_instance" not in st.session_state:
        # stx.CookieManager 초기화 (처음 로딩 시 1회)
        st.session_state["cookie_manager_instance"] = stx.CookieManager(key="eers_cookie_manager")
    
    cookie_manager = st.session_state["cookie_manager_instance"]
    token = cookie_manager.get(cookie="eers_auth_token")

    if token and not st.session_state.get("logged_in_success"):
        # 쿠키가 있고 아직 로그인되지 않은 경우 (세션 복원)
        st.session_state["logged_in_success"] = True
        st.session_state["target_email"] = token
        st.session_state["auth_stage"] = "complete"
        
        return True
    
    return st.session_state.get("logged_in_success", False)

def render_auth_ui():
    """로그인 및 인증 단계를 사이드바에 렌더링"""
    
    # 0. 쿠키 매니저 및 로그인 상태 확인
    is_logged_in = check_auth_cookie()
    cookie_manager = st.session_state.get("cookie_manager_instance")
    # Safety Check: 만약 인스턴스가 없다면 (매우 드문 경우)
    if not cookie_manager:
        st.error("Cookie Manager 초기화 오류. 앱을 새로고침하십시오.")
        return
    
    st.session_state.setdefault("generated_code", None)
    st.session_state.setdefault("code_timestamp", None)
    # 로그인 성공 후 auth_stage는 'complete'로 설정되어야 함
    st.session_state.setdefault("auth_stage", "input_email" if not is_logged_in else "complete")

    # 로그인 성공 상태
    if is_logged_in:
        email_full = st.session_state.get("target_email", "")
        st.markdown(f"**로그인:** <span style='text-decoration:none;'>{email_full}</span>", unsafe_allow_html=True)

        if st.button("로그아웃", key="sidebar_logout_btn_success", type="secondary", use_container_width=True):
            logout()
 
        return

    # 1. 이메일 입력 단계
    if st.session_state["auth_stage"] == "input_email":
        st.caption("사내 메일(@kepco.co.kr)로 인증 코드를 발송합니다.")
        
        email_id = st.text_input(
            "메일 ID",
            key="sidebar_email_id_input",
            placeholder="메일 ID를 입력하세요"
        )
        st.text_input(
            "도메인",
            value="@kepco.co.kr",
            disabled=True,
            key="sidebar_email_domain"
        )

        full_email = f"{email_id}@kepco.co.kr" if email_id else ""
        
        submitted = st.button("인증코드 발송", type="primary", use_container_width=True, key="sidebar_send_code")

        if submitted:
            if not email_id:
                st.error("❌ 이메일을 입력하세요.")
            else:
                code = "".join(random.choices(string.digits, k=6))
                
                st.session_state["generated_code"] = code
                st.session_state["target_email"] = full_email
                st.session_state["code_timestamp"] = datetime.now()

                with st.spinner("메일 발송 중..."):
                    if send_verification_email(full_email, code):
                        st.toast("📧 인증코드 발송 완료! 인증코드 입력 단계로 이동합니다.")
                        st.session_state["auth_stage"] = "verify_code"
                        st.rerun() # 상태 변경 후 UI 갱신
                    else:
                        st.error("메일 발송 실패! (SMTP 설정 및 로그 확인)")
        return
        


    # -------------------------------
    # ⭐⭐⭐ 여기다 넣는다!! ⭐⭐⭐
    # -------------------------------
    def _verify_code_submit():
        """엔터 입력 시 자동 로그인"""
        code_input = st.session_state.get("sidebar_code_input_verify", "")
        generated = st.session_state.get("generated_code", "")
        cookie_manager = st.session_state["cookie_manager_instance"]

        time_limit_sec = 300
        elapsed = datetime.now() - st.session_state["code_timestamp"]
        if elapsed.total_seconds() > time_limit_sec:
            st.error("⏰ 인증 시간이 만료되었습니다. 다시 시도해주세요.")
            st.session_state["auth_stage"] = "input_email"
            st.rerun()
            return

        if code_input == generated:
            st.session_state["logged_in_success"] = True
            st.session_state["auth_stage"] = "complete"

            expire_date = datetime.now() + timedelta(days=180)
            cookie_manager.set(
                "eers_auth_token",
                st.session_state["target_email"],
                expires_at=expire_date
            )
            st.toast("로그인 성공!", icon="✅")
            st.rerun()
        else:
            st.error("❌ 인증코드가 일치하지 않습니다.")


    # 2. 인증코드 입력 단계
    if st.session_state["auth_stage"] == "verify_code":
        time_limit_sec = 300 # 5분
        time_limit = timedelta(seconds=time_limit_sec)
        elapsed = datetime.now() - st.session_state["code_timestamp"]
        remaining = max(0, int(time_limit.total_seconds() - elapsed.total_seconds()))
        
        # 만료 처리
        if remaining <= 0:
            st.error("⏰ 인증 시간이 만료되었습니다. 다시 시도해주세요.")
            st.session_state["auth_stage"] = "input_email"
            # st.rerun() # 만료 후 바로 Rerun 대신, 다음 1초 Rerun에 맡깁니다.
            return

        # 타이머 및 안내
        st.info(f"📩 **{st.session_state.get('target_email', '주소 미확인')}**로 발송된 6자리 인증코드를 입력하세요.")
        st.markdown(f"**⏳ 남은 시간: ** <span style='color:red; font-weight:bold;'>{remaining}초</span>", unsafe_allow_html=True)

        code_input = st.text_input(
        "인증코드 6자리",
        max_chars=6,
        key="sidebar_code_input_verify",
        label_visibility="collapsed",
        on_change=_verify_code_submit  # 👈 Enter 입력 시 자동 실행
    )
        col_login, col_back = st.columns(2)
        
        login_btn = col_login.button("로그인", type="primary", use_container_width=True, key="sidebar_login_btn_verify")
        back_btn = col_back.button("이메일 다시 입력", key="sidebar_back_btn")
        
        # --- 버튼 클릭 로직 ---
        if back_btn:
            st.session_state["auth_stage"] = "input_email"
            st.rerun()
            return

        if login_btn:
            if remaining <= 0:
                 st.error("⏰ 인증 시간이 만료되었습니다. 다시 시도해주세요.")
                 st.session_state["auth_stage"] = "input_email"
                 st.rerun()
            elif code_input == st.session_state["generated_code"]:
                # 로그인 성공 처리
                st.session_state["logged_in_success"] = True
                st.session_state["auth_stage"] = "complete" 
                
                expire_date = datetime.now() + timedelta(days=180)
                cookie_manager.set( 
                    "eers_auth_token",
                    st.session_state["target_email"],
                    expires_at=expire_date
                )
                st.toast("로그인 성공!", icon="✅")
                st.rerun() 
            else:
                st.error("❌ 인증코드가 일치하지 않습니다.")
                
        # 타이머 갱신을 위해 1초마다 강제 재실행
        time.sleep(1)
        st.rerun()




# =========================================================
# 자동 업데이트 스케줄러 (백그라운드 스레드)
# =========================================================

@st.cache_resource
def start_auto_update_scheduler():
    def scheduler_loop():
        last_run_hour = -1
        while True:
            now = datetime.now()
            
            if now.hour in [8, 12, 19]:
                if now.minute == 0 and now.hour != last_run_hour:
                    try:
                        # 💥 변경: print 대신 logger.info 사용
                        logger.info(f"[Auto-Sync] {now} - 자동 업데이트 시작")
                        
                        target_date_str = now.strftime("%Y%m%d")
                        
                        for stage in STAGES_CONFIG.values():
                            fetch_data_for_stage(target_date_str, stage)
                            
                        _set_last_sync_datetime_to_meta(now)
                        
                        # 캐시 클리어
                        _get_new_item_counts_by_source_and_office.clear()
                        load_data_from_db.clear()
                        
                        # 💥 변경: print 대신 logger.info 사용
                        logger.info(f"[Auto-Sync] {now} - 자동 업데이트 완료")
                        last_run_hour = now.hour
                        
                    except Exception as e:
                        # 💥 변경: print 대신 logger.error 사용
                        logger.error(f"[Auto-Sync] 오류 발생: {e}")
            
            time.sleep(30)

    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    logger.info(">>> 자동 업데이트 스케줄러 스레드가 시작되었습니다.") # 💥 변경



# =========================================================
# 1. 세션 상태 및 DB 세션
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
    ss.setdefault("admin_auth", False) # 관리자 인증
    ss.setdefault("logged_in_success", False) # 일반 로그인
    ss.setdefault("df_data", pd.DataFrame())
    ss.setdefault("total_items", 0)
    ss.setdefault("total_pages", 1)
    ss.setdefault("data_initialized", False)
    ss.setdefault("route_page", "공고 조회 및 검색")
    ss.setdefault("view_mode", "카드형") # 💡 [수정] 초기값 "카드형"
    ss.setdefault("selected_notice", None)
    ss.setdefault("is_updating", False)
    ss.setdefault("show_login_dialog", False) # 로그인 다이얼로그 상태
    st.session_state.setdefault("show_login_form", False) 
    st.session_state.setdefault("auth_stage", "input_email") # 인증 단계 초기화


def get_db_session():
    # PostgreSQL은 create_all을 여러번 호출해도 문제 없음
    Base.metadata.create_all(engine)
    return SessionLocal()



# 신규 건수 집계
@st.cache_data(ttl=300)
def _get_new_item_counts_by_source_and_office() -> dict:
    session = get_db_session()
    if not session: return {}
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
            # 복수관할 처리 로직 유지
            if "/" in office_name:
                parts = [p.strip() for p in office_name.split("/") if p.strip()]
                for part in parts:
                    counts.setdefault(part, {"G2B": 0, "K-APT": 0})
                    source_key = "K-APT" if source == "K-APT" else "G2B"
                    counts[part][source_key] += count // len(parts)
            else:
                counts.setdefault(office_name, {"G2B": 0, "K-APT": 0})
                source_key = "K-APT" if source == "K-APT" else "G2B"
                counts[office_name][source_key] += count

        total_g2b = sum(v.get("G2B", 0) for v in counts.values())
        total_kapt = sum(v.get("K-APT", 0) for v in counts.values())
        counts["전체"] = {"G2B": total_g2b, "K-APT": total_kapt}
        return counts
    except Exception as e:
        print(f"신규 건수(소스별) 집계 오류: {e}")
        return {}
    finally:
        session.close()

# =========================================================
# 2. 데이터 로딩 (공고 조회)
# =========================================================

@st.cache_data(ttl=600, show_spinner="데이터를 조회 중...")
def load_data_from_db(
    office, source, start_date, end_date, keyword, only_cert, include_unknown, page,
):
    session = get_db_session()
    if not session: return pd.DataFrame(), 0 # 더미 반환

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    query = session.query(Notice).filter(
        Notice.notice_date.between(start_date_str, end_date_str)
    )

    if source == "나라장터": query = query.filter(Notice.source_system == "G2B")
    elif source == "K-APT": query = query.filter(Notice.source_system == "K-APT")

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
        is_dlvr_no_format = bool(re.match(r"^[A-Z0-9]{10,}$", keyword_text.replace("-", "").upper()))
        
        if is_dlvr_no_format:
            normalized = keyword_text.replace("-", "").upper()
            query = query.filter(Notice.detail_link.like(f"%dlvrreq:{normalized}%"))
        else:
            terms = [t.strip() for t in keyword_text.split() if t.strip() and not t.startswith("-")]
            if terms:
                query = query.filter(or_(*[or_(*[c.ilike(f"%{term}%") for c in cols]) for term in terms]))

    total_items = query.count()
    offset = (page - 1) * ITEMS_PER_PAGE
    rows = query.order_by(Notice.notice_date.desc(), Notice.id.desc()).offset(offset).limit(ITEMS_PER_PAGE).all()
    
    # 데이터 프레임 변환 로직 유지
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
            "⭐": "★" if n.is_favorite else "☆",
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
            "IS_FAVORITE": bool(n.is_favorite),
            "IS_NEW": is_new,
        })

    df = pd.DataFrame(data)
    session.close()
    return df, total_items

def search_data():
    if engine and not inspect(engine).has_table("notices"):
        Base.metadata.create_all(engine)

    # 💡 [수정] 페이지 초기화
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

    total_pages = (
        max(1, math.ceil(st.session_state.total_items / ITEMS_PER_PAGE))
        if st.session_state.total_items > 0
        else 1
    )
    st.session_state.total_pages = total_pages
    st.session_state["data_initialized"] = True # 데이터 조회 완료 표시
    # st.rerun() # 불필요한 reru 방지
    

# =========================================================
# 3. 상세 보기 / 즐겨찾기 (수정)
# =========================================================

def toggle_favorite(notice_id: int):
    """즐겨찾기 토글 (로그인 필요)"""
    if not st.session_state.get("logged_in_success"):
        st.error("❌ 즐겨찾기 기능은 로그인 후 사용할 수 있습니다.")
        return

    session = get_db_session()
    if not session: return # DB 세션이 없을 경우 종료

    try:
        n = session.query(Notice).filter(Notice.id == notice_id).one_or_none()
        if n:
            n.is_favorite = not bool(n.is_favorite)
            if not n.is_favorite:
                n.status = ""
                n.memo = ""
            session.commit()
            st.toast("즐겨찾기 상태가 변경되었습니다.")

            # 즐겨찾기 변경 후 데이터 다시 로드
            load_data_from_db.clear()
            _get_new_item_counts_by_source_and_office.clear()

            # 현재 페이지의 데이터를 다시 조회
            search_data_no_rerun() 
            st.rerun() # UI 갱신

    except Exception as e:
        st.error(f"즐겨찾기 변경 중 오류: {e}")
        session.rollback()
    finally:
        session.close()

# 💡 search_data 함수를 비동기 호출 없이 세션 상태만 업데이트하는 헬퍼 함수
def search_data_no_rerun():
    if engine and not inspect(engine).has_table("notices"):
        Base.metadata.create_all(engine)
    try:
        df, total_items = load_data_from_db(
            st.session_state["office"], st.session_state["source"],
            st.session_state["start_date"], st.session_state["end_date"],
            st.session_state["keyword"], st.session_state["only_cert"],
            st.session_state["include_unknown"], st.session_state["page"],
        )
        st.session_state.df_data = df
        st.session_state.total_items = total_items
        st.session_state.total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))
    except Exception as e:
        print(f"데이터 조회 중 오류 (no rerun): {e}")


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
    kapt_code = rec.get("KAPT_CODE")
    if not kapt_code:
        st.error("단지 코드가 없어 상세 정보를 조회할 수 없습니다.")
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
            
            gb.configure_selection(
                selection_mode="single", use_checkbox=False, pre_selected_rows=[0]
            )
            
            gb.configure_column("순번", width=60, cellStyle={'textAlign': 'center'})
            gb.configure_column("품명", width=200)
            
            grid_options = gb.build()

            grid_response = AgGrid(
                df, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=250, theme="alpine", allow_unsafe_jscode=False, key=f"dlvr_grid_{req_no}"
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

# [수정] 캐시 데코레이터(@st.cache_data)를 삭제하여 항상 DB에서 최신 조회
def _get_recipients_from_db(offices: list[str]) -> list[dict]:
    session = get_db_session()
    target_offices = [o for o in offices if o and o != "전체"]

    recipients = []
    # 1. 활성 상태(is_active=True)인 수신자만 조회
    q = session.query(MailRecipient).filter(MailRecipient.is_active == True)
    
    # 2. 선택된 사업소 필터링
    if "전체" not in offices and target_offices:
        q = q.filter(MailRecipient.office.in_(target_offices))

    for r in q.order_by(MailRecipient.email).all():
        if r.email:
            recipients.append(
                {
                    "email": r.email.strip().lower(),
                    "office": r.office,
                    "name": r.name or "",
                }
            )
    session.close() # 세션 닫기 추가 권장
    return recipients


def _filter_unknown(items: list[dict], include_unknown: bool):
    if include_unknown:
        return items

    filtered_items = []
    UNKNOWN_STR = {
        "관할불명",
        "미확인",
        "미정",
        "불명",
        "unknown",
        "UNKNOWN",
        "확인필요",
        "확인 필요",
        "관할지사확인요망",
    }

    for item in items:
        office_val = item.get("assigned_office", "").strip()
        if "/" in office_val:
            continue
        if any(u.lower() in office_val.lower() for u in UNKNOWN_STR):
            continue
        filtered_items.append(item)
    return filtered_items


def _query_items_for_period(session, start: date, end: date, office: str):
    q = session.query(Notice).filter(
        Notice.notice_date >= start.isoformat(),
        Notice.notice_date <= end.isoformat(),
    )
    if office and office != "전체":
        q = q.filter(
            or_(
                Notice.assigned_office == office,
                Notice.assigned_office.like(f"{office}/%"),
                Notice.assigned_office.like(f"%/{office}"),
                Notice.assigned_office.like(f"%/{office}/%"),
            )
        )

    q = q.order_by(Notice.notice_date.desc())
    rows = q.all()
    items = []
    for r in rows:
        items.append(
            {
                "source_system": r.source_system or "",
                "assigned_office": r.assigned_office or "",
                "stage": r.stage or "",
                "project_name": r.project_name or "",
                "client": r.client or "",
                "address": (r.address or ""),
                "phone_number": r.phone_number or "",
                "model_name": r.model_name or "",
                "quantity": r.quantity or 0,
                "is_certified": r.is_certified or "",
                "notice_date": r.notice_date or "",
                "detail_link": r.detail_link or "",
            }
        )
    return items


def _save_history(
    session,
    office,
    subject,
    period,
    to_list,
    total_count,
    attach_name,
    preview_html,
):
    h = MailHistory(
        office=office,
        subject=subject,
        period_start=period[0].isoformat(),
        period_end=period[1].isoformat(),
        to_list=";".join(to_list),
        total_count=total_count,
        attach_name=attach_name,
        preview_html=preview_html,
    )
    session.add(h)
    session.commit()



# =========================================================
# 6-1. 팝업(모달) 래퍼 함수 추가
# =========================================================

@st.dialog("상세 정보", width="large")
def popup_detail_panel(rec: dict):
    show_detail_panel(rec)


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

                b1, b2 = st.columns(2)
                is_logged_in = st.session_state.get("logged_in_success", False)
                star_label = "★ 즐겨찾기" if rec.get("IS_FAVORITE") else "☆ 즐겨찾기"

                with b1:
                    if st.button(star_label, key=f"fav_card_{rec['id']}", use_container_width=True, disabled=not is_logged_in):
                        toggle_favorite(rec["id"])

                with b2:
                    if st.button("🔍 상세", key=f"detail_card_{rec['id']}", use_container_width=True):
                        popup_detail_panel(rec)


def render_notice_table(df):
    st.markdown("### 📋 공고 목록")

    if df.empty:
        st.info("표시할 공고가 없습니다.")
        return None

    df_disp = df.copy()
    df_disp["⭐"] = df_disp["IS_FAVORITE"]
    df_disp.insert(0, "상세", "🔍") 

    def format_title(row):
        title = row["사업명"]
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

        if source == "K-APT":
            if is_real_new: prefixes.append("🔵 [NEW]")
        elif is_existing_new:
            prefixes.append("🔴 [NEW]")

        return f"{' '.join(prefixes)} {title}" if prefixes else title

    df_disp["사업명"] = df_disp.apply(format_title, axis=1)

    visible_cols = [
        "id", "상세", "⭐", "순번", "구분", "사업소", "단계", "사업명", 
        "기관명", "소재지", "연락처", "모델명", "수량", "고효율 인증 여부", "공고일자"
    ]
    final_cols = [c for c in visible_cols if c in df_disp.columns]

    # ----------------------------------
    # 2. AgGrid 옵션 설정 (편집 및 체크박스 활성화)
    # ----------------------------------
    gb = GridOptionsBuilder.from_dataframe(df_disp[final_cols])
    
    is_logged_in = st.session_state.get("logged_in_success", False)
    
    gb.configure_column(
        "⭐", 
        width=60, 
        editable=is_logged_in, # 💡 [수정] 로그인 시에만 편집 가능
        cellStyle={'textAlign': 'center'},
        type=['booleanColumn', 'centerAligned']
    )

    gb.configure_selection("single", use_checkbox=False, pre_selected_rows=[])
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    gb.configure_column("id", hide=True)
    gb.configure_column("상세", width=50, cellStyle={'textAlign': 'center'}, pinned='left')
    gb.configure_column("순번", width=70, cellStyle={'textAlign': 'center'})
    gb.configure_column("구분", width=90, cellStyle={'textAlign': 'center'})
    gb.configure_column("단계", width=90, cellStyle={'textAlign': 'center'})
    gb.configure_column("사업명", width=450)
    
    # 💡 [추가] 상세 보기 버튼 클릭 처리
    js_func = JsCode("""
        function(params) {
            if (params.column.colId === '상세' && params.data.id) {
                // '상세' 컬럼 클릭 시 해당 행의 ID를 이용하여 Streamlit에 전달
                Streamlit.set
            }
        }
    """)
    
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_disp[final_cols], gridOptions=gridOptions, 
        update_mode=GridUpdateMode.VALUE_CHANGED, 
        data_return_mode=DataReturnMode.AS_INPUT, fit_columns_on_grid_load=False,
        height=350, theme='streamlit'
    )

    # ----------------------------------
    # 4. 선택 및 토글 로직 처리 (데이터 비교)
    # ----------------------------------
    edited_df_raw = grid_response.get('data') 
    
    # 1) 즐겨찾기 토글 감지 및 처리
    if is_logged_in and edited_df_raw is not None and not edited_df_raw.empty:
        df_comp = df[['id', 'IS_FAVORITE']].copy()
        df_comp = df_comp.rename(columns={'IS_FAVORITE': 'IS_FAVORITE_original'})

        merged_df = pd.merge(df_comp, edited_df_raw[['id', '⭐']], on='id', how='inner')
        merged_df = merged_df.rename(columns={'⭐': '⭐_edited'})
        changed_rows = merged_df[merged_df['IS_FAVORITE_original'] != merged_df['⭐_edited']]
        
        if not changed_rows.empty:
            changed_id = changed_rows.iloc[0]['id']
            toggle_favorite(int(changed_id)) 
            return None 

    # 2) 행 선택 감지 및 반환 (상세 보기)
    selected_rows = grid_response.get('selected_rows')
    target_row_dict = None

    if hasattr(selected_rows, "empty"): 
        if not selected_rows.empty:
            target_row_dict = selected_rows.iloc[0].to_dict()
    elif isinstance(selected_rows, list) and len(selected_rows) > 0:
        target_row_dict = selected_rows[0]

    if target_row_dict:
        try:
            sel_id = target_row_dict.get("id")
            # 💡 [수정] '상세' 버튼이 눌렸는지 확인 (선택된 행의 '상세' 컬럼 값으로 확인)
            if target_row_dict.get("상세") == "🔍":
                original_series = df[df["id"] == sel_id].iloc[0]
                return original_series.to_dict() 
        except Exception:
            return None

    return None

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

    st.title("💡 대구본부 EERS 업무 지원 시스템")
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


# =========================================================
# 8. 로그인 필요 페이지들 (기존 코드 유지)
# =========================================================

def favorites_page():
    st.title("⭐ 관심 고객 관리")
    
    col_filter, _ = st.columns([1, 3])
    with col_filter:
        selected_office = st.selectbox("사업소 필터", OFFICES, key="fav_office_select")

    st.info("체크 해제 후 '상태/메모 저장' 버튼을 누르면 관심 고객에서 해제됩니다.")

    session = get_db_session()
    if not session:
        st.error("데이터베이스 연결 오류.")
        return

    query = session.query(Notice).filter(Notice.is_favorite == True)

    if selected_office != "전체":
        query = query.filter(
            or_(
                Notice.assigned_office == selected_office,
                Notice.assigned_office.like(f"{selected_office}/%"),
                Notice.assigned_office.like(f"%/{selected_office}"),
                Notice.assigned_office.like(f"%/{selected_office}/%"),
            )
        )

    favs = query.order_by(Notice.notice_date.desc()).all()
    session.close()

    if not favs:
        st.warning(f"'{selected_office}' 사업소에 관심 고객으로 등록된 공고가 없습니다.")
        return

    data = []
    STATUSES = ["", "미접촉", "전화", "메일안내", "접수", "지급", "보류", "취소"]

    for n in favs:
        data.append({
            "id": n.id, "⭐": True,
            "사업소": (n.assigned_office or "").replace("/", "\n"),
            "사업명": n.project_name or "", "기관명": n.client or "",
            "공고일자": _as_date(n.notice_date).isoformat() if n.notice_date else "",
            "상태": n.status or "", "메모": n.memo or "",
            "DETAIL_LINK": n.detail_link or "", "KAPT_CODE": n.kapt_code or "",
            "SOURCE": n.source_system,
        })

    df_favs = pd.DataFrame(data)

    edited_df = st.data_editor(
        df_favs.drop(columns=["DETAIL_LINK", "KAPT_CODE", "SOURCE"]),
        column_config={
            "⭐": st.column_config.CheckboxColumn("⭐", help="클릭하여 관심 고객 해제", default=True), 
            "상태": st.column_config.SelectboxColumn("상태", options=STATUSES, required=True),
            "메모": st.column_config.TextColumn("메모", default="", max_chars=200),
            "사업명": st.column_config.Column("사업명", width="large"),
            "사업소": st.column_config.Column("사업소", width="medium"),
            "id": None,
        },
        hide_index=True, key="fav_editor", use_container_width=True,
    )

    col_save, col_export, col_spacer = st.columns([1.5, 1.5, 10])

    if col_save.button("상태/메모 저장"):
            session = get_db_session()
            if not session:
                st.error("DB 연결 오류")
                return
            updates = 0
            favorites_set = 0
            unfavorites = 0
            try:
                for _, row in edited_df.iterrows():
                    n = session.query(Notice).filter(Notice.id == row["id"]).one()
                    
                    is_status_memo_changed = (n.status != row["상태"] or n.memo != row["메모"])
                    is_favorite_changed = (n.is_favorite != row["⭐"])
                    
                    if is_status_memo_changed:
                        n.status = row["상태"]
                        n.memo = row["메모"]
                        updates += 1
                    
                    if is_favorite_changed:
                        n.is_favorite = row["⭐"]
                        if row["⭐"]: favorites_set += 1
                        else: unfavorites += 1

                    if is_status_memo_changed or is_favorite_changed:
                        session.add(n)

                session.commit()
                
                msg = []
                if updates > 0: msg.append(f"{updates}건의 상태 및 메모가 저장되었습니다.")
                if favorites_set > 0: msg.append(f"{favorites_set}건이 관심 고객으로 설정되었습니다.")
                if unfavorites > 0: msg.append(f"{unfavorites}건이 관심 고객에서 해제되었습니다.")

                if msg: st.success(" ".join(msg))
                else: st.info("변경된 내용이 없습니다.")
                    
                load_data_from_db.clear()
                st.rerun()

            except Exception as e:
                st.error(f"저장 중 오류 발생: {e}")
                session.rollback()
            finally:
                session.close()

    @st.cache_data
    def convert_df_to_excel(df):
        output = BytesIO()
        df.drop(columns=["id", "⭐"], errors="ignore").to_excel(output, index=False, engine="openpyxl")
        return output.getvalue()

    col_export.download_button(
        label="엑셀로 저장",
        data=convert_df_to_excel(edited_df),
        file_name="eers_favorites.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================================================
# 9. 관리자 전용 페이지들 (기존 코드 유지)
# =========================================================

def mail_send_page():

    # 시작일/종료일 초기값 셋팅 (지난주 월~일)
    def _set_last_week_default():
        today = date.today()
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)

        st.session_state.setdefault("mail_start", last_monday)
        st.session_state.setdefault("mail_end", last_sunday)

    _set_last_week_default()


    st.title("✉️ 메일 발송")

    # (2) 이전 발송 결과 표시
    if "mail_send_result" in st.session_state:
        result = st.session_state.pop("mail_send_result")
        if result["type"] == "success":
            st.success(result["message"])
        else:
            st.error(result["message"])

    # ============================
    # ① 사업소 선택 / 기간 설정
    # ============================

    col_office, col_period = st.columns(2)

    with col_office:
        st.subheader("발송 사업소")
        office_options = ["전체"] + [o for o in OFFICES if o not in MAIL_EXCLUDE_OFFICES]

        selected_offices = st.multiselect(
            "사업소 선택 (복수 선택 가능)",
            options=office_options,
            default=["전체"],
            key="mail_office_select",
        )

        include_unknown = st.checkbox(
            "관할불명/복수관할 항목 포함", key="mail_include_unknown"
        )

    with col_period:
        st.subheader("발송 기간 설정")
        btn_col1, btn_col2, _ = st.columns(3)

        def set_last_week():
            today = date.today()
            this_monday = today - timedelta(days=today.weekday())
            last_monday = this_monday - timedelta(days=7)
            last_sunday = last_monday + timedelta(days=6)
            st.session_state["mail_start"] = last_monday
            st.session_state["mail_end"] = last_sunday

        def set_last_month():
            today = date.today()
            first_this = date(today.year, today.month, 1)
            last_prev = first_this - timedelta(days=1)
            first_prev = date(last_prev.year, last_prev.month, 1)
            st.session_state["mail_start"] = first_prev
            st.session_state["mail_end"] = last_prev

        if btn_col1.button("지난 주 (월~일)"):
            set_last_week()
        if btn_col2.button("지난 달"):
            set_last_month()

        if "mail_start" not in st.session_state:
            st.session_state["mail_start"] = DEFAULT_END_DATE - timedelta(days=7)
        if "mail_end" not in st.session_state:
            st.session_state["mail_end"] = DEFAULT_END_DATE

        start_date = st.date_input("시작일", st.session_state["mail_start"], key="mail_start")
        end_date = st.date_input("종료일", st.session_state["mail_end"], key="mail_end")

    st.markdown("---")

    # ============================
    # ② 수신자 목록 표시
    # ============================

    recipients_data = _get_recipients_from_db(selected_offices)
    email_list = [r["email"] for r in recipients_data]

    if not email_list:
        st.error("❌ 선택한 사업소에 등록된 메일 수신자가 없습니다.\n수신자 관리 메뉴에서 등록해 주세요.")
        st.stop()

    with st.expander(f"수신자 목록 ({len(email_list)}명)", expanded=False):
        if recipients_data:
            df_rec = pd.DataFrame(recipients_data).rename(columns={
                "office": "사업소",
                "name": "담당자명",
                "email": "이메일"
            })
            st.dataframe(
                df_rec,
                hide_index=True,
                use_container_width=True,
                column_order=df_rec.columns.tolist(),
                column_config={col: st.column_config.Column(disabled=True) for col in df_rec.columns},
            )
        else:
            st.warning("선택된 사업소에 수신자가 없습니다. '수신자 관리'에서 등록해주세요.")

    st.markdown("---")

    # ============================
    # ③ 메일 미리보기 생성 버튼
    # ============================

    if st.button("📄 메일 미리보기", key="preview_btn"):
        if start_date > end_date:
            st.error("시작일은 종료일보다 늦을 수 없습니다.")
            st.stop()

        session = get_db_session()
        mail_preview_data = {}

        with st.spinner("메일 내용 준비 중..."):

            year = start_date.year
            year_start, year_end = date(year, 1, 1), date(year, 12, 31)

            for office in selected_offices:
                items_period = _query_items_for_period(session, start_date, end_date, office)
                items_period = _filter_unknown(items_period, include_unknown)
                items_annual = _query_items_for_period(session, year_start, year_end, office)

                if not items_period and not items_annual:
                    continue

                subject = build_subject(office, (start_date, end_date), len(items_period))
                body, attach_name, attach_html, preview = build_body_html(
                    office, (start_date, end_date), items_period, items_annual
                )

                mail_preview_data[office] = {
                    "subject": subject,
                    "html_body": body,
                    "to_list": _get_recipients_from_db([office]),
                    "attach_name": attach_name,
                    "attach_html": attach_html,
                    "items_period": items_period,
                }

        if not mail_preview_data:
            st.info("발송할 내용이 없습니다.")
            st.stop()

        st.session_state["mail_preview_data"] = mail_preview_data
        st.success("미리보기가 준비되었습니다!")
        st.rerun()

    # ============================
    # ④ 미리보기 탭 표시
    # ============================

    if "mail_preview_data" in st.session_state:
        mpd = st.session_state["mail_preview_data"]

        st.subheader("발송 전 최종 확인")
        tab_titles = list(mpd.keys())
        tabs = st.tabs(tab_titles)

        for i, office in enumerate(tab_titles):
            data = mpd[office]
            with tabs[i]:
                st.markdown(f"**제목:** {data['subject']}")
                st.markdown(f"**수신자:** {', '.join(r['email'] for r in data['to_list'])}")
                st.markdown(f"**신규 공고 건수:** {len(data['items_period'])}건")
                st.markdown("---")
                st.markdown("**본문 미리보기 (HTML)**")
                st.components.v1.html(data["html_body"], height=400, scrolling=True)

        st.markdown("---")
        st.info("미리보기를 확인하셨다면 발송을 진행하세요.")

        # ============================
        # ⑤ 최종 발송 버튼
        # ============================

        if st.button("📨 최종 발송 실행 (SMTP)", key="final_send_btn"):
            st.session_state["_do_final_send"] = True
            st.rerun()

    # ============================
    # ⑥ 실제 발송 실행
    # ============================

    if st.session_state.get("_do_final_send"):
        mpd = st.session_state["mail_preview_data"]
        sent, failed = [], {}

        with st.spinner("메일 발송 중..."):
            for office, data in mpd.items():
                try:
                    # 💡 수정된 부분: mailer.py의 send_mail 함수가 
                    # SMTP 설정값들을 인수로 받도록 변경되었다고 가정하고 추가합니다.
                    send_mail(
                        to_list=[r["email"] for r in data["to_list"]],
                        subject=data["subject"],
                        html_body=data["html_body"],
                        attach_name=data["attach_name"],
                        attach_html=data["attach_html"],
                        # ---------------------------------------------
                        # 🔥 추가된 인수
                        mail_from=MAIL_FROM, 
                        smtp_host=MAIL_SMTP_HOST, 
                        smtp_port=MAIL_SMTP_PORT, 
                        mail_user=MAIL_USER, 
                        mail_pass=MAIL_PASS,
                        # ---------------------------------------------
                    )
                    sent.append(office)
                except Exception as e:
                    msg = str(e)

                    # SMTP 사용자 친화적 메시지 변환
                    if "553" in msg:
                        msg = "수신자 이메일 주소가 올바르지 않습니다. 수신자 관리에서 확인해 주세요."
                    elif "535" in msg:
                        msg = "SMTP 로그인 실패 — 메일 서버 아이디/비밀번호를 확인하세요."
                    elif "Timed out" in msg:
                        msg = "메일 서버 연결이 지연되었습니다. 네트워크 상태를 확인하세요."

                    failed[office] = msg
        st.session_state["_do_final_send"] = False
        st.session_state.pop("mail_preview_data", None)

        result_msg = []

        # --------------------
        # 성공 부분
        # --------------------
        if sent:
            result_msg.append(f"✅ 발송 성공: {', '.join(sent)}")

        # --------------------
        # 실패 부분 — 줄바꿈 적용
        # --------------------
        if failed:
            fail_lines = [f"{office}: {err}" for office, err in failed.items()]
            fail_block = "\n".join(fail_lines)

            result_msg.append(f"❌ 발송 실패:\n\n{fail_block}")

        # --------------------
        # 최종 메시지 저장
        # --------------------
        st.session_state["mail_send_result"] = {
            "type": "success" if sent else "error",
            "message": "\n".join(result_msg),
        }


        st.rerun()



# =========================================================
# 수신자 관리 관련 헬퍼
# =========================================================

def load_rows_by_office_from_db():
    data = {}
    session = get_db_session()
    if not session:
        return {}
    try:
        rows = (
            session.query(MailRecipient)
            .order_by(MailRecipient.office, MailRecipient.email)
            .all()
        )
        for r in rows:
            data.setdefault(r.office, []).append(
                {
                    "use": bool(getattr(r, "is_active", True)),
                    "office": getattr(r, "office", ""),
                    "name": getattr(r, "name", ""),
                    "id": r.email.split("@")[0] if getattr(r, "email", "") else "",
                    "domain": r.email.split("@")[1] if "@" in getattr(r, "email", "") else "",
                }
            )
    except Exception as e:
        st.error(f"DB 로드 오류: {e}")
    finally:
        session.close()
    return data

# =========================================================
# 수신자 관리 페이지 (UI)
# =========================================================



#=========================================================
# 수신자 관리 저장 헬퍼
# =========================================================

def save_rows_by_office_to_db(df_editor):

    def _normalize(val, default=""):
        if val is None:
            return default
        if isinstance(val, list):
            if not val:
                return default
            val = val[0]
        if pd.isna(val):
            return default
        return str(val)

    session = get_db_session()
    if not session:
        st.error("DB 오류: 세션 생성 실패")
        return

    failed_rows = []
    saved_count = 0

    try:
        # 기존 데이터 삭제
        session.query(MailRecipient).delete()
        session.flush()

        for idx, row in df_editor.iterrows():

            office = _normalize(row["사업소명"])
            name = _normalize(row["담당자명"])
            local = _normalize(row["이메일 ID"])
            domain = "@kepco.co.kr"

            # -----------------------------
            # ⚠ 유효성 검사
            # -----------------------------
            error_msg = None

            if office == "":
                error_msg = "사업소명이 비어 있습니다."
            elif local == "":
                error_msg = "이메일 ID가 비어 있습니다."
            elif " " in local:
                error_msg = "이메일 ID에 공백이 들어있습니다."

            if error_msg:
                failed_rows.append(f"{idx+1}번째 행 오류: {error_msg}")
                continue

            email = f"{local}@kepco.co.kr"

            use_val = row["발송대상"]
            is_active = (
                use_val if isinstance(use_val, bool)
                else str(use_val).lower() in ["1", "true", "yes"]
            )

            # DB 저장
            session.add(
                MailRecipient(
                    office=office,
                    email=email.lower(),
                    name=name,
                    is_active=is_active,
                )
            )
            saved_count += 1

        # 저장 개수 확인
        if saved_count == 0:
            session.rollback()
            st.error("❌ 저장된 수신자가 없습니다. 아래 오류를 확인하세요.")
            return

        session.commit()

        # 성공 메시지
        st.success(f"✅ 총 {saved_count}명 저장 완료!")

        # 실패한 행도 알려주기
        if failed_rows:
            st.warning("⚠ 일부 행은 저장되지 않았습니다:")
            for err in failed_rows:
                st.warning(err)

        time.sleep(0.7)
        st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"🔥 예외 오류 발생: {e}")

    finally:
        session.close()

def mail_manage_page():
    st.title("👤 수신자 관리")

    if not st.session_state.admin_auth:
        st.error("관리자만 접근 가능합니다.")
        return

    # 사업소 목록
    all_office_list = [o for o in OFFICES if o != "전체"]

    # -------------------------
    # DB → DataFrame 변환
    # -------------------------
    raw = load_rows_by_office_from_db()
    rows = []

    for office, items in raw.items():
        for r in items:
            rows.append({
                "발송대상": bool(r["use"]),       # ✔ 첫 번째 선택 컬럼 → 삭제 / 이걸 주 컬럼으로
                "사업소명": office,
                "담당자명": r["name"],
                "이메일 ID": r["id"],
                "도메인": r["domain"] or "kepco.co.kr",
            })

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame({
            "발송대상": [],
            "사업소명": [],
            "담당자명": [],
            "이메일 ID": [],
            "도메인": [],
        })

    df["도메인"] = df["도메인"].replace("", "kepco.co.kr")
    df["도메인"] = df["도메인"].fillna("kepco.co.kr")


    # @ 컬럼 추가
    df["@"] = "@"

    # 타입 강제 변환
    df["발송대상"] = df["발송대상"].astype(bool)
    df["사업소명"] = df["사업소명"].astype(str)
    df["담당자명"] = df["담당자명"].astype(str)
    df["이메일 ID"] = df["이메일 ID"].astype(str)
    df["@"] = df["@"].astype(str)
    df["도메인"] = df["도메인"].astype(str)


    # -------------------------
    # Data Editor
    # -------------------------

    column_order = [
        "발송대상",
        "사업소명",
        "담당자명",
        "이메일 ID",
        "@",
        "도메인",
    ]


    edited_df = st.data_editor(
        df,
        column_config={
            "발송대상": st.column_config.CheckboxColumn("발송대상", default=True),
            "사업소명": st.column_config.SelectboxColumn("사업소명", options=all_office_list),
            "담당자명": st.column_config.TextColumn("담당자명"),
            "이메일 ID": st.column_config.TextColumn("이메일 ID"),
            "@": st.column_config.Column("@", disabled=True),
            "도메인": st.column_config.Column("도메인", disabled=True),
        },
        column_order=column_order, 
        num_rows="dynamic",
        hide_index=True,
        key="recipient_editor",
        use_container_width=True,
    )

    if st.button("주소록 최종 저장", type="primary"):
        save_rows_by_office_to_db(edited_df)



def data_sync_page():
    st.title("🔄 데이터 업데이트")
    if not st.session_state.admin_auth:
        st.error("데이터 업데이트는 관리자만 사용할 수 있습니다.")
        return

    # ... (기존 데이터 업데이트 로직 유지)
    last_dt = _get_last_sync_datetime_from_meta()
    last_txt = last_dt.strftime("%Y-%m-%d %H:%M") if last_dt else "기록 없음"
    st.info(f"마지막 API 호출 일시: **{last_txt}**")
    st.markdown("---")

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

    if st.button("선택 기간 업데이트 시작", type="primary", key="start_sync_btn"):
        if start_date > end_date:
            st.error("시작일은 종료일보다 늦을 수 없습니다.")
            st.stop()
        if (end_date - start_date).days >= 92:
            st.error("조회 기간은 최대 92일(3개월)까지만 가능합니다.")
            st.stop()
        
        st.session_state["is_updating"] = True



        st.subheader("📊 데이터 수집 진행률")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 💡 로그 메시지를 Streamlit UI에 표시할 컨테이너 (st.info 사용)
        log_placeholder = st.container()



        dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        stages_to_run = list(STAGES_CONFIG.values())
        total_steps = len(dates) * len(stages_to_run)
        current_step = 0

        # 로그 메시지 저장용 리스트
        sync_logs = []

        try:
            for d in dates:
                disp_date = d.strftime("%Y-%m-%d")
                for stage in stages_to_run:
                    name = stage.get("name", "Unknown Stage")
                    status_text.markdown(f"**현재:** `{disp_date} / {name}`")

                    try:
                        fetch_data_for_stage(d.strftime("%Y%m%d"), stage)
                        sync_logs.append(f"✔ [{disp_date}] {name} 완료")
                    except Exception as e:
                        error_msg = f"❌ [{disp_date}] {name} 오류 : {e}"
                        sync_logs.append(error_msg)
                        logger.error(error_msg) # 💡 콘솔 로그에 오류 기록

                    current_step += 1
                    pct = int(current_step / total_steps * 100)
                    progress_bar.progress(pct / 100)
                    status_text.markdown(f"**진행률:** {pct}% ({current_step}/{total_steps})")

                    # 로그 업데이트: 매 단계마다 컨테이너를 비우고 다시 씁니다.
                    with log_placeholder:
                        st.info("\n".join(sync_logs))

            status_text.success("🎉 전체 작업 완료!") #

            progress_bar.progress(1.0)
        
            _set_last_sync_datetime_to_meta(datetime.now())
            load_data_from_db.clear()
            _get_new_item_counts_by_source_and_office.clear()
            st.success("데이터 수집이 완료되었습니다. 상단 '공고 조회 및 검색'에서 다시 조회해 주세요.")
            st.session_state["is_updating"] = False
            st.rerun()

        except Exception as global_e:
            status_text.error(f"⚠️ 동기화 작업 중 치명적인 오류 발생: {global_e}")
            logger.error(f"Global Sync Error: {global_e}", exc_info=True) # 추가 로깅
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
                        "id": n.id, "⭐": "★" if n.is_favorite else "☆",
                        "구분": "K-APT" if n.source_system == "K-APT" else "나라장터",
                        "사업소": (n.assigned_office or "").replace("/", " "),
                        "단계": n.stage or "", "사업명": n.project_name or "",
                        "기관명": n.client or "", "소재지": n.address or "",
                        "연락처": fmt_phone(n.phone_number or ""), "모델명": n.model_name or "",
                        "수량": str(n.quantity or 0),
                        "고효율 인증 여부": _normalize_cert(n.is_certified),
                        "공고일자": date_str, "DETAIL_LINK": n.detail_link or "",
                        "KAPT_CODE": n.kapt_code or "", "IS_FAVORITE": bool(n.is_favorite),
                        "IS_NEW": False
                    })
                
                df_day = pd.DataFrame(data)
                
                rec = render_notice_table(df_day)
                
                if rec: popup_detail_panel(rec)
            else:
                st.info("해당 조건의 데이터가 없습니다.")


# =========================================================
# 7. 관리자 인증 / 사이드바 / 전체 앱 실행 (최종 수정)
# =========================================================

def admin_auth_modal():
    """관리자 인증 모달 (일반 로그인 상태에서 추가 인증)"""
    
    if not st.session_state.get("logged_in_success", False):
        return

    if st.session_state.admin_auth:
        st.success("✅ 관리자 인증 완료")
        if st.sidebar.button("인증 해제", key="btn_admin_logout_sidebar"):
            st.session_state.admin_auth = False
            st.toast("관리자 권한이 해제되었습니다.")
            st.rerun()
        return
    

    # -------------------------
    # 🔥 엔터 입력 시 자동 인증되도록 콜백 추가
    # -------------------------
    def _admin_submit():
        pwd = st.session_state.get("sidebar_admin_password_input", "")
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_auth = True
            st.toast("✅ 인증 성공! 관리자 권한이 활성화되었습니다.", icon="✅")
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")


    with st.sidebar.expander("🔑 관리자 추가 인증"):
        password = st.text_input(
            "비밀번호를 입력하세요:", type="password", key="sidebar_admin_password_input",
            label_visibility="collapsed"
        )
        
        if st.button("인증", key="btn_admin_login_sidebar", use_container_width=True):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.toast("✅ 인증 성공! 관리자 권한이 활성화되었습니다.", icon="✅")
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")


def eers_app():
    st.set_page_config(
        page_title="EERS 업무 지원 시스템",
        layout="wide",
        page_icon="💡",
        initial_sidebar_state="expanded",
    )

    if "cookie_manager_instance" not in st.session_state:
        st.session_state["cookie_manager_instance"] = stx.CookieManager(key="eers_cookie_manager")

    init_session_state()
    
    # [쿠키 기반 로그인 상태 복구]
    cookie_manager = st.session_state["cookie_manager_instance"]
    auth_cookie = cookie_manager.get("eers_auth_token")

    if auth_cookie and not st.session_state.get("logged_in_success", False):
        st.session_state["logged_in_success"] = True
        st.session_state["target_email"] = auth_cookie
        st.toast("쿠키를 통해 자동 로그인되었습니다.", icon="👋")
        # 💡 [수정] 로그인 성공 시 auth_stage 초기화
        st.session_state["auth_stage"] = "complete"

    #start_auto_update_scheduler()

    # [사이드바 구성]
    with st.sidebar:
        st.header("EERS 업무 지원 시스템")
        
        is_logged_in = st.session_state.get("logged_in_success", False)

        if is_logged_in:
            # 로그인된 상태
            email_full = st.session_state.get("target_email", "")
            st.markdown(f"**로그인:** <span style='text-decoration:none;'>{email_full}</span>", unsafe_allow_html=True)


            if st.button("로그아웃", key="sidebar_logout_btn", type="secondary", use_container_width=True):
                logout()
                st.rerun()

        else:
            # 로그인 안된 상태 → 로그인 버튼만 보이게 함
            if st.button("🔑 로그인", key="sidebar_login_btn", type="primary", use_container_width=True):
                st.session_state["show_login_dialog"] = True
                st.session_state["auth_stage"] = "input_email"
                st.rerun()

        # 로그인 UI 표시 (버튼 아래)
        if st.session_state.get("show_login_dialog", False) and not st.session_state.get("logged_in_success"):
            render_auth_ui()        

        # 💡 [핵심] 로그인 상태에 따른 메뉴 분기
        is_logged_in = st.session_state.get("logged_in_success", False)
        is_admin = st.session_state.get("admin_auth", False)
        
        # 관리자 인증 (로그인 상태에서만 표시)
        admin_auth_modal()
        
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


        # ---------------------------
        # 로그인 후 메뉴
        # ---------------------------
        if is_logged_in:
            st.markdown("### 👤 업무 담당자 기능")
            render_menu_button("관심 고객 관리")
            render_menu_button("데이터 현황")


        # ---------------------------
        # 관리자 인증 후 메뉴
        # ---------------------------
        if is_admin:
            st.markdown("### 🛠 관리자 기능")
            render_menu_button("메일 발송")
            render_menu_button("수신자 관리")
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

    # [페이지 라우팅]
    page = st.session_state.route_page
    if page == "공고 조회 및 검색":
        main_page()
    elif page == "관심 고객 관리" and is_logged_in:
        favorites_page()
    elif page == "메일 발송" and is_admin:
        mail_send_page()
    elif page == "수신자 관리" and is_admin:
        mail_manage_page()
    elif page == "데이터 업데이트" and is_admin:
        data_sync_page()
    elif page == "데이터 현황" and is_logged_in:
        data_status_page()
    else:
        # 로그인 필요 기능에 미로그인 상태로 접근 시 (혹시 모를 오류 대비)
        main_page()

    # [로그인 다이얼로그 표시]
    # 💡 [수정] show_login_dialog가 True일 때 팝업 호출
    if st.session_state.get("show_login_dialog", False) and not st.session_state.get("logged_in_success"):
            # login_dialog() # <-- render_auth_ui()로 통합되어 삭제됨
            pass

if __name__ == "__main__":
    if engine and not inspect(engine).has_table("notices"):
        Base.metadata.create_all(engine)
    eers_app()