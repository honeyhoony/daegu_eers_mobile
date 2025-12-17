# Writing the cleaned & adjusted collect_data.py with your requested changes.
from typing import List, Dict, Optional, Tuple, Callable
import sys
import json, time, requests
# === 학교 힌트 로더 & 학교명 추출 ===
import os, unicodedata
from functools import lru_cache
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy.dialects.postgresql import insert as pg_insert # <--- 함수 맨 위(import 영역)에 추가
from database import Base, Notice, engine  # noqa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import streamlit as st
try:
    from bjd_mapper import get_bjd_name
    HAS_BJD_MAPPER = True
except ImportError:
    print("[Warning] bjd_mapper.py 파일을 찾을 수 없어 주소 변환이 제한됩니다.")
    HAS_BJD_MAPPER = False
import re
from difflib import SequenceMatcher
from contextlib import contextmanager

@contextmanager
def get_session():
    s = Session()
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()




# -------------------------------------------------------
# CONFIG LOADER (로컬 config.py + Streamlit secrets 자동 지원)
# -------------------------------------------------------

# 1) Streamlit import 시도
try:
    import streamlit as st
    HAS_ST = True
except:
    HAS_ST = False
    class _DummySecrets(dict):
        def get(self, key, default=None):
            return default
    st = type("dummy", (), {"secrets": _DummySecrets()})

# 2) 로컬 config.py import 시도
try:
    import config as _local_config
    HAS_LOCAL_CONFIG = True
except:
    _local_config = None
    HAS_LOCAL_CONFIG = False

# 3) 최종 config getter (Fly.io 대응)
def _cfg(key: str, default=None):
    """
    우선순위
    1. 환경변수 (Fly.io, Docker, 운영)
    2. 로컬 config.py
    3. Streamlit Cloud secrets.toml
    4. default
    """

    # 1️⃣ Fly.io / Docker
    if key in os.environ:
        return os.environ.get(key)

    # 2️⃣ 로컬 개발환경
    if HAS_LOCAL_CONFIG and hasattr(_local_config, key):
        return getattr(_local_config, key)

    # 3️⃣ Streamlit Cloud
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# =========================
# API Keys (Fly.io secrets)
# =========================

NARA_SERVICE_KEY = _cfg("NARA_SERVICE_KEY")
KEA_SERVICE_KEY  = _cfg("KEA_SERVICE_KEY")
KAPT_SERVICE_KEY = _cfg("KAPT_SERVICE_KEY")

if not NARA_SERVICE_KEY:
    raise RuntimeError("NARA_SERVICE_KEY 누락")
if not KEA_SERVICE_KEY:
    raise RuntimeError("KEA_SERVICE_KEY 누락")
if not KAPT_SERVICE_KEY:
    raise RuntimeError("KAPT_SERVICE_KEY 누락")



def normalize_model_for_compare(m: str) -> str:
    """불필요 기호 제거하고 소문자 정규화"""
    if not m:
        return ""
    m = m.lower().strip()
    m = re.sub(r"[^a-z0-9]", "", m)  # 영숫자만 남기기
    return m

def model_similarity(a: str, b: str) -> float:
    """모델명 유사도 계산 (0~1)"""
    a_n = normalize_model_for_compare(a)
    b_n = normalize_model_for_compare(b)
    if not a_n or not b_n:
        return 0
    return SequenceMatcher(None, a_n, b_n).ratio()



LOG_EXCLUDES = False   # 타지역/제외 로그 출력 여부

def print_exclude_once(base_notice: dict, client_name: Optional[str], addr_or_mall: Optional[str]):
    if not LOG_EXCLUDES:
        return
    key = base_notice.get("detail_link") or f"{base_notice.get('stage')}|{base_notice.get('project_name')}|{client_name}"
    if not PRINT_DEDUP or key not in _SEEN_EXCLUDE_KEYS:
        if PRINT_DEDUP:
            _SEEN_EXCLUDE_KEYS.add(key)
        print(f"  [❌ 제외 (타 지역)] {client_name or ''} - {addr_or_mall or ''}")



EXCLUDE_LOG_MAX = 0 if not LOG_EXCLUDES else 50

# 파일 상단 유틸
import os, json, unicodedata
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip())

def load_school_map() -> dict[str, str]:
    # 1) 파이썬 모듈 우선
    try:
        from client_hints_schools import CLIENT_HINTS_SCHOOLS as _S  # :contentReference[oaicite:1]{index=1}
        # A/B 형태도 있을 수 있으니 첫 지사만 미리 정리
        return {_norm(k): _norm(v.split("/")[0]) for k, v in _S.items()}
    except Exception:
        pass

    # 2) JSON(있으면)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(base_dir, "client_hints_schools.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {_norm(k): _norm(v.split("/")[0]) for k, v in data.items()}
    except Exception:
        pass

    # 3) 없으면 빈 dict
    return {}

def load_client_hints_schools() -> dict[str, str]:
    """
    client_hints_schools.json을 읽어 {학교명: 지사} 딕셔너리로 반환.
    파일이 없거나 파싱 실패 시 빈 dict.
    """
    # 경로는 프로젝트 구조에 맞게 조정
    # 예: 스크립트와 같은 폴더라면:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "client_hints_schools.json")

    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 키/값 정규화
        return {_norm(k): (v.split("/")[0].strip()) for k, v in data.items()}
    except Exception:
        return {}
from typing import Tuple

def decide_office_and_address_by_apt_or_bjd(kapt_code: str = "", bjd_code: str = "", addr_text: str = "") -> Tuple[str, str]:
    """
    우선순위:
      1) apt_list(단지코드)에서 address/office를 우선 사용
      2) apt_list에 office가 없으면 bjd_code(또는 addr)로 사업소 추론
      3) address가 비어있으면 bjd_mapper로 주소 보강
    """
    # 1) apt_list 우선
    csv_addr, csv_office, csv_bjd = lookup_apt_by_code(kapt_code)
    picked_addr = (csv_addr or addr_text or "").strip()
    chosen_office = (csv_office or "").strip()
    bjd_for_fallback = (csv_bjd or bjd_code or "").strip()

    # 2) office 결정 (apt_list가 비어있으면 bjd 기준)
    if not chosen_office:
        chosen_office = _assign_office_from_bjd_code(bjd_code=bjd_for_fallback, addr_text=picked_addr)

    # 3) 주소 보강 (apt_list/원문 없으면 bjd_mapper로)
    if not picked_addr:
        picked_addr = resolve_address_from_bjd(bjd_code=bjd_for_fallback, addr_text=picked_addr)

    return chosen_office, picked_addr


def process_kapt_item(it: dict, page_stage: str = "입찰공고") -> dict | None:
    """
    K-APT 단일 아이템 처리:
    - apt_list(단지코드) 우선 → 실패 시 bjd_mapper 폴백으로 주소/지사 결정
    - 표시주소(display_address) 생성(도로명 + (동) 보강)
    - Notice dict 생성
    """
    # 0) 안전 추출
    kapt_code   = _as_text(it.get("aptCode") or it.get("kaptCode"))
    bjd_code    = _as_text(it.get("bjdCode") or it.get("bidArea") or it.get("bjd_code"))
    addr_raw    = _as_text(it.get("roadAddr") or it.get("addr") or it.get("bidAddr"))
    project_name = _as_text(it.get("bidTitle") or it.get("projectName") or "공고명 없음")
    client_name  = _as_text(it.get("bidKaptname") or it.get("client") or project_name)
    amount_txt   = _as_text(it.get("amount") or "")
    notice_dt    = to_ymd(it.get("bidRegdate") or it.get("noticeDate"))

    # 1) 주소/사업소 결정 (apt_list 우선 → bjd 폴백)
    # K-APT 기본정보(연락처/주소 보강) 즉시 조회
    basic = fetch_kapt_basic_info(kapt_code) if kapt_code else None
    phone_mgmt = _extract_kapt_phone(basic)  # ← 관리사무소 전화
    # 주소/사업소 결정
    office, addr_core = decide_office_and_address_by_apt_or_bjd(
        kapt_code=kapt_code, bjd_code=bjd_code, addr_text=(addr_raw or (basic or {}).get("doroJuso") or (basic or {}).get("kaptAddr") or "")
    )

    # 2) 표시주소 생성(도로명 + (동) 보강)
    #    _compose_display_addr는 dict에서 addr/roadAddr/bjdCode/as1/as2/as3 등을 참고합니다.
    compose_input = dict(it)
    compose_input.update({
        "addr": addr_core or addr_raw,
        "roadAddr": addr_raw,
        "bjdCode": bjd_code,
        "kaptCode": kapt_code,
    })
    display_addr = _compose_display_addr(compose_input)
    it["display_address"] = display_addr  # 필요 시 다른 곳에서도 활용 가능

    # 3) 상세링크 생성(없으면 안전 폴백)
    bid_num = _as_text(it.get("bidNum"))
    if bid_num:
        detail_link = f"https://www.k-apt.go.kr/bid/bidDetail.do?no={bid_num}"
    else:
        detail_link = "https://www.k-apt.go.kr/bid/bidList.do"

    # 4) 기본 Notice 뼈대
    base = _build_base_notice(
        stage=page_stage,
        biz_type=_as_text(it.get("codeClassifyType1") or "기타"),
        project_name=project_name,
        client=client_name,
        phone=phone_mgmt,             # ✅ 관리사무소 전화 즉시 반영
        model="",
        qty=0,
        amount=amount_txt,
        is_cert="확인필요",
        notice_date=notice_dt,
        detail_link=detail_link,
        source="K-APT",
        kapt_code=kapt_code
    )

    # 5) 주소/사업소 최종 반영
    addr_final = display_addr or addr_core or addr_raw
    base["office"]    = office
    base["mall_addr"] = addr_final

    # 6) Notice dict 마감
    n = finalize_notice_dict(base, None, addr_final, client_name)
    return n


def fetch_pages_parallel(url, params_list):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # 동시 5스레드
        futures = [executor.submit(http_get_json, url, p) for p in params_list]
        for f in as_completed(futures):
            results.append(f.result())
    return results



def bulk_upsert_notices(notices):
    if not notices:
        return
    
    session.begin()
    try:
        from sqlalchemy.dialects.postgresql import insert # collect_data.py 상단에 추가

        stmt = insert(Notice).values(notices)
        # 사용자가 직접 관리하는 is_favorite, status, memo는 업데이트에서 제외
        update_cols = {
            col.name: col
            for col in stmt.excluded
            if col.name not in ['id', 'is_favorite', 'status', 'memo']
        }

        stmt = stmt.on_conflict_do_update(
            # Notice 모델의 __table_args__에 정의된 UniqueConstraint 이름과 일치해야 함
            index_elements=["source_system", "detail_link", "model_name", "assigned_office"],
            set_=update_cols
        )

        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"  [Error] Bulk upsert 실패: {e}")





@lru_cache(maxsize=5000)
def get_usr_info_cached(dminstt_code):
    return get_full_address_from_usr_info(dminstt_code)

import random
def safe_get(url, params):
    for i in range(3):
        try:
            return http_get_json(url, params)
        except:
            time.sleep(0.5 * (2 ** i) + random.random())
    return None


USE_KEA_CHECK = True  # 일단 끕니다. (필요할 때만 True)

# =========================
# 공통 설정
# =========================
API_HOST = "apis.data.go.kr"
API_SCHEME = "http"   # 내부망/방화벽 환경 고려


# === K-APT 최소 로그 포맷 (주소를 끝에 붙여서 출력) ===
def _fmt_tail(addr: str) -> str:
    addr = _as_text(addr).strip()
    return f" - {addr}" if addr else ""

def log_kapt_excluded(name: str, addr: str = ""):
    if not LOG_EXCLUDES:
        return
    print(f"[❌ 제외 (타 지역)] {_as_text(name)}{_fmt_tail(addr)}")

def log_kapt_pending(office: str, name: str, addr: str = ""):
    print(f"[🧺 저장 대기] {office} / {name}" + (f" - {addr}" if addr else ""))

def log_kapt_saved(office: str, name: str, addr: str = ""):
    print(f"[✅ 저장 완료] {_as_text(office)} / {_as_text(name)}{_fmt_tail(addr)}")

def log_kapt_bulk_saved(n: int):
    print(f"[✅ 일괄 저장] {int(n)}건")


# =========================
# 성능 최적화 옵션
# =========================
USE_NAME_BASED_USRINFO = False   # 기관명 기반 UsrInfo 보조조회 사용 여부
VERBOSE = False                  # 디버깅 로그 출력 여부

def log(msg: str):
    if VERBOSE:
        print(msg)


# =========================
# 로그 중복 방지 (detail_link 기준 1회만)
# =========================
PRINT_DEDUP = True
_SEEN_EXCLUDE_KEYS = set()


# 엔드포인트
ORDER_PLAN_LIST_PATH = "/1230000/ao/OrderPlanSttusService/getOrderPlanSttusListThng"
BID_LIST_PATH        = "/1230000/ao/PubDataOpnStdService/getDataSetOpnStdBidPblancInfo"
CNTRCT_LIST_PATH     = "/1230000/ao/CntrctInfoService/getCntrctInfoListThng"
DLVR_LIST_PATH       = "/1230000/at/ShoppingMallPrdctInfoService/getDlvrReqInfoList"
USR_INFO_PATH        = "/1230000/ao/UsrInfoService/getDminsttInfo"
# [추가] 공동주택(K-APT)
KAPT_BID_LIST_PATH = "/1613000/ApHusBidPblAncInfoOfferServiceV2/getPblAncDeSearchV2"
KAPT_PRIVATE_CONTRACT_PATH = "/1613000/ApHusPrvCntrNoticeInfoOfferServiceV2/getRegDeSearchV2"
# [추가] K-APT 단지정보/유지관리이력 API 엔드포인트
KAPT_BASIC_INFO_PATH = "/1613000/AptBasisInfoServiceV4/getAphusBassInfoV4"
KAPT_DETAIL_INFO_PATH = "/1613000/AptBasisInfoServiceV4/getAphusDtlInfoV4"
KAPT_MAINTENANCE_PATH = "/1613000/ApHusMntMngHistInfoOfferServiceV2/getElctyExtgElvtrMntncHistInfoSearchV2"


def _kapt_items_safely(data) -> list[dict]:
    """
    K-APT 응답을 어떤 형태로 받더라도 list[dict]로 안전 정규화.
    허용 케이스:
      - {"response":{"body":{"items": ...}}}
      - {"response":{"body":{"item": ...}}}
      - {"body":{"items": ...}} / {"body":{"item": ...}}
      - 최상위가 곧바로 list/dict 인 경우
    """
    if data is None:
        return []

    # 1) 최상위가 바로 list면 dict만 추려서 반환
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    # 2) 최상위 dict에서 계층적으로 파고들기
    if isinstance(data, dict):
        # 가장 일반형
        body = ((data.get("response") or {}).get("body") or {})
        for key in ("items", "item"):
            cont = body.get(key)
            if cont is not None:
                return _as_items_list(cont)

        # 변형: 최상위에 바로 items/item
        for key in ("items", "item", "list", "data"):
            cont = data.get(key)
            if cont is not None:
                return _as_items_list(cont)

        # 여차하면 dict 전체를 단건 취급
        return [data] if data else []

    # 기타 타입은 무시
    return []


def _as_items_list(obj) -> list[dict]:
    """
    dict/list/None/단건 혼재 응답을 안전하게 list[dict]로 정규화.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        # 리스트 원소가 dict가 아닐 수도 있으니 dict만 필터
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # body.items 혹은 단건 dict
        # 흔한 케이스: {"item":[{...},{...}]}
        for key in ("items","item","list","data"):
            v = obj.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                return [v]
        return [obj]
    # 나머지는 문자열 등 → 빈 목록
    return []



def api_url(path: str) -> str:
    return f"{API_SCHEME}://{API_HOST}{path}"

# HTTP 세션/재시도
SESSION = requests.Session()
SESSION.trust_env = True
SESSION.headers.update({
    "User-Agent": "EERS-Collector/2.2 (+https://g2b.go.kr)",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
})
_retries = Retry(total=3, backoff_factor=0.4, status_forcelist=(429, 500, 502, 503, 504))
_adapter = HTTPAdapter(
    max_retries=_retries,
    pool_connections=100,   # 추가
    pool_maxsize=100        # 추가
)
SESSION.mount("http://", _adapter)
SESSION.mount("https://", _adapter)
DEFAULT_TIMEOUT = (5, 20)  # (connect, read)

# =========================
# 유틸
# =========================
# In collect_data.py (at the very bottom of the file)
# [ADD] 나라장터 전용: 텍스트에서 '○○초/중/고/대학(교)' 학교명만 뽑기

# 디버그 출력 (VERBOSE일 때만)
def _debug(msg: str):
    if VERBOSE:
        print(msg)

# 합계 요약 한 줄
def _print_total_summary(total: int, *, tag: str | None = None):
    pages = (int(total) + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 0
    if tag:
        #print(f"- 총 {total}건 / {pages}p ({tag})")
        print(f"- 총 {total}건")
    else:
        #print(f"- 총 {total}건 / {pages}p")
        print(f"- 총 {total}건")


# ===== 로그 헬퍼 (나라장터 시그니처 정렬) =====
def _print_data_none():
    print("  - 데이터 없음")

def _print_bulk_saved(n: int, prefix: str = ""):
    # prefix가 비어있지 않으면 "  [✅ prefix 일괄 저장] N건"
    if prefix:
        print(f"  [✅ {prefix} 일괄 저장] {int(n)}건")
    else:
        print(f"  [✅ 일괄 저장] {int(n)}건")

def _debug(msg: str):
    if VERBOSE:
        print(msg)


def _extract_school_name(*parts: str) -> str | None:
    txt = " ".join([p for p in parts if p]).strip()
    if not txt:
        return None
    # 괄호 제거 + 공백 정리
    txt = re.sub(r"[\[\(（].*?[\]\)）]", " ", txt)
    txt = " ".join(txt.split())

    SUFFIX = r"(?:초등학교|중학교|고등학교|대학교|대학|마이스터고등학교)"
    # 전체에서 '…학교' 덩어리 후보들 추출
    candidates = re.findall(rf"([가-힣0-9A-Za-z·\-\s]+?{SUFFIX})", txt)
    if not candidates:
        return None

    # 가장 오른쪽 후보 선택
    cand = sorted(set(candidates), key=lambda s: (txt.rfind(s), len(s)))[-1].strip()

    # 🔧 여기서 마지막 토큰만 남기기: (교육청 + 학교) → (학교)
    m = re.search(rf"([^\s]+?{SUFFIX})\s*$", cand)
    if m:
        cand = m.group(1).strip()

    # 유니코드 정규화(가끔 섞여 들어오는 특수공백 대비)
    import unicodedata
    cand = unicodedata.normalize("NFKC", cand)
    return cand or None

# [ADD] 나라장터 전용: 학교명 기반 지사 배정 (CLIENT_HINTS_SCHOOLS 우선)
def _assign_office_by_school_name(client_name: str, project_name: str) -> str | None:
    """
    - client_hints_schools.py(외부 사전)를 최우선 사용
    - 없거나 미스매치면 CLIENT_HINTS(통합 사전)에서 '학교' 키만 폴백
    - 비교는 모두 NFKC 정규화 + 양방향 부분일치
    """
    school = _extract_school_name(client_name, project_name)
    if not school:
        return None
    s_norm = _norm(school)

    # 1순위: 별도 학교 사전 (client_hints_schools.py)
    try:
        from client_hints_schools import CLIENT_HINTS_SCHOOLS as _S
        # 키/값 모두 정규화, 값은 "A/B"면 첫 지사만
        S = {_norm(k): _norm(v.split("/")[0]) for k, v in _S.items()}
        for k in sorted(S.keys(), key=len, reverse=True):
            # 양방향 부분일치 허용 (교육청+학교 형태 등 긴/짧은 양쪽 커버)
            if s_norm == k or s_norm in k or k in s_norm:
                return S[k]
    except Exception:
        pass

    # 2순위: 통합 힌트 사전(CLASSIC) - '학교' 키만 대상
    try:
        C = CLIENT_HINTS  # collect_data.py 등에서 선언/병합되어 있다고 가정
        for k in sorted(C.keys(), key=len, reverse=True):
            if "학교" not in k:
                continue
            k_norm = _norm(k)
            if s_norm == k_norm or s_norm in k_norm or k_norm in s_norm:
                # 값도 "A/B"면 첫 지사만
                return _norm((C[k] or "").split("/")[0])
    except Exception:
        pass

    return None

def _as_text(x) -> str:
    """리스트/숫자/None 등도 안전하게 문자열로 변환."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, list):
        return " ".join(_as_text(v) for v in x)
    try:
        # dict 등은 JSON 문자열화 (ensure_ascii=False로 한글 유지)
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)
    

@lru_cache(maxsize=1)
def _load_school_hints() -> dict:
    """
    client_hints_schools.py / client_hints_schools.json 병합 로드
    우선순위: PY → JSON (PY가 있으면 우선)
    """
    hints = {}

    # 1) .py 로드
    try:
        from client_hints_schools import CLIENT_HINTS_SCHOOLS as _PY_HINTS
        if isinstance(_PY_HINTS, dict):
            hints.update(_PY_HINTS)
    except Exception:
        try:
            # 일부 프로젝트는 키 이름이 CLIENT_HINTS 인 경우가 있음
            from client_hints_schools import CLIENT_HINTS as _PY_HINTS2
            if isinstance(_PY_HINTS2, dict):
                hints.update(_PY_HINTS2)
        except Exception:
            pass

    # 2) .json 로드 (있으면 병합: PY에 없는 키만 추가)
    try:
        base_dir = os.path.dirname(__file__)
        jpath = os.path.join(base_dir, "client_hints_schools.json")
        if os.path.isfile(jpath):
            with open(jpath, "r", encoding="utf-8") as f:
                j = json.load(f)
                if isinstance(j, dict):
                    for k, v in j.items():
                        hints.setdefault(k, v)
    except Exception:
        pass

    return hints

def extract_school_name(client_name: str) -> str:
    """
    고객명 문자열에서 '학교명'만 힌트 테이블 키를 기준으로 추출.
    - 가장 긴 키가 부분일치 하는 것을 우선 채택(정확도↑)
    - '(... )' 꼬리표 제거, 공백/중복 공백 정리
    """
    if not client_name:
        return ""

    # 괄호 등 끝부분 부가표기 제거
    name = re.sub(r"\s*\([^)]*\)\s*$", "", client_name).strip()
    name_no_space = re.sub(r"\s+", "", name)

    hints = _load_school_hints()
    if not hints:
        return name  # 힌트가 없으면 원문 반환

    # 키를 길이 내림차순으로 정렬 → 가장 긴 키부터 매칭
    keys = sorted(hints.keys(), key=lambda k: len(k), reverse=True)

    best = ""
    for key in keys:
        k = key.strip()
        if not k:
            continue
        k_no_space = re.sub(r"\s+", "", k)
        # 부분 포함(공백 무시 비교 포함)
        if (k in name) or (k_no_space in name_no_space):
            best = k
            break

    return best or name  # 최종 실패시 원문 반환

def office_by_school_hint(school_name: str) -> str:
    """
    추출된 '학교명'으로 사업소 힌트 조회.
    값에 'A/B'처럼 슬래시가 있으면 첫 항목 사용.
    """
    if not school_name:
        return ""
    hints = _load_school_hints()
    hit = hints.get(school_name.strip())
    if not hit:
        return ""
    return hit.split("/")[0].strip()


def _has_dong_level_str(a: str) -> bool:
    return bool(re.search(r"(동|읍|면|리)\b", a or ""))

def _narrow_office_with_basic_info(assigned: str, kapt_code: str, addr_txt: str, bjd_code: str):
    """
    A/B(복수관할) 이거나 주소에 동/읍/면 레벨이 없으면
    K-APT 기본정보로 주소/법정동코드 보강 후 관할 재판정.
    """
    assigned = _as_text(assigned).strip()
    addr_txt = _as_text(addr_txt)
    bjd_code = _as_text(bjd_code)

    try:
        need_narrow = ("/" in assigned) or (not _has_dong_level_str(addr_txt))
        if not need_narrow or not kapt_code:
            return assigned, addr_txt, bjd_code

        basic = fetch_kapt_basic_info(kapt_code) or {}
        addr2 = (basic.get("doroJuso") or basic.get("kaptAddr") or addr_txt or "").strip()
        bjd2  = str(basic.get("bjdCode") or bjd_code or "").strip()

        # 주소가 아직도 비면 bjd_mapper로 보강
        if not addr2 and bjd2:
            try:
                from bjd_mapper import get_bjd_name
                addr2 = (get_bjd_name(bjd2) or "").strip()
            except Exception:
                pass

        reassigned = _assign_office_from_bjd_code(bjd2, addr2)
        # 성공적으로 단일 관할로 내려가면 교체
        if reassigned and "/" not in reassigned and not reassigned.startswith("관할"):
            return reassigned, addr2 or addr_txt, bjd2 or bjd_code
        # 실패 시 기존 값 유지
        return assigned, addr_txt, bjd_code
    except Exception:
        return assigned, addr_txt, bjd_code



def _has_dong_level(a: str) -> bool:
    a = _as_text(a)
    return bool(__import__("re").search(r"(동|읍|면|리)\b", a))


def _to_int(v) -> int:
    """'1,234' / '10.0' / ' 10 ' 등도 안전 변환."""
    try:
        if v is None: return 0
        s = str(v).strip().replace(",", "")
        if not s: return 0
        return int(float(s))
    except Exception:
        return 0


def cleanup_session():
    """전역 세션을 닫아 데이터베이스 연결을 해제합니다."""
    global session
    if session:
        try:
            session.close()
            print("[DB] Worker session closed.")
        except Exception as e:
            print(f"[DB] Worker session close error: {e}")



def _handle_broad_keyword_case(client_name: Optional[str], addr: Optional[str], base_notice: dict) -> bool:
    name = client_name or ""
    if not name:
        return False

    # ✅ 부산/해운대구 등 타권역이 명시돼 있으면 즉시 제외
    other_hits = any(kw in name for kw in ["부산", "해운대구", "해운대"])  # 필요한 경우 더 추가
    target_hits = any(kw in name for kw in ["대구", "경상북도", "경북", "포항", "경주", "경산", "김천", "영천", "칠곡", "성주", "청도", "고령", "영덕"])
    if other_hits and not target_hits:
        return False

    offices_to_assign = None
    # 🔁 이전: if "대구" in name: ...
    # ✅ 변경: '대구', '대구시', '대구광역시'를 '단어'로만 인식
    if _contains_token(name, ["대구", "대구시", "대구광역시"]):
        offices_to_assign = DAEGU_OFFICES
    # 🔁 이전: elif "포항" in name:
    elif _contains_token(name, ["포항", "포항시"]):
        offices_to_assign = ["포항지사", "북포항지사"]

    if offices_to_assign:
        offices_str = "/".join(offices_to_assign)
        n = dict(base_notice)
        n["assigned_office"] = offices_str
        n["address"] = addr or ""
        upsert_notice(n)
        session.commit()
        print(f"  [⚠️ 저장 (복수 관할)] {offices_str} / {n.get('client')}")
        return True

    # 나머지 광역 키워드는 행정구역 단어 경계로 판단
    for keyword, office in BROAD_KEYWORD_OFFICE_MAP.items():
        if _contains_token(name, [keyword]):
            n = dict(base_notice)
            n["assigned_office"] = office
            n["address"] = addr or ""
            upsert_notice(n)
            session.commit()
            print(f"  [✅ 저장 완료] {n.get('assigned_office')} / {n.get('client')}")
            return True

    return False

# === K-APT 키워드 필터: config → ENV → 기본값 ===
import os, json, re

def _get_conf_list(attr_name: str, env_name: str, default_list):
    # 1) config.py에 리스트가 있으면 사용
    try:
        import config as _cfg
        if hasattr(_cfg, attr_name):
            v = getattr(_cfg, attr_name)
            if isinstance(v, (list, tuple, set)): return list(v)
            if isinstance(v, str) and v.strip(): return [v.strip()]
    except Exception:
        pass
    # 2) 환경변수(JSON 배열 또는 콤마 구분)
    s = os.getenv(env_name, "")
    if s.strip():
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)): return list(parsed)
        except Exception:
            return [t.strip() for t in s.split(",") if t.strip()]
    # 3) 기본값
    return list(default_list or [])

# 기본 포함/제외 (원하면 바꾸세요)
_KAPT_INC_RAW = _get_conf_list("KAPT_INCLUDE_KEYWORDS", "KAPT_INCLUDE_KEYWORDS",
                               ["승강기", "led", "변압기", "인버터", "펌프", "/엘리베이터|인버터|모터|제어반/"])
_KAPT_EXC_RAW = _get_conf_list("KAPT_EXCLUDE_KEYWORDS", "KAPT_EXCLUDE_KEYWORDS",
                               ["조경", "제설", "/도장|외벽/"])

def _compile_patterns(patterns):
    out = []
    for p in (patterns or []):
        s = str(p or "").strip()
        if not s: continue
        try:
            if len(s) >= 2 and s[0] == "/" and s[-1] == "/":
                out.append(("regex", re.compile(s[1:-1], re.IGNORECASE)))
            else:
                out.append(("text", s.lower()))
        except Exception:
            out.append(("text", s.lower()))
    return out

_INC_PAT = _compile_patterns(_KAPT_INC_RAW)
_EXC_PAT = _compile_patterns(_KAPT_EXC_RAW)

def _match_patterns(text: str, pats):
    if not pats: return False
    t = (text or "").lower()
    for kind, obj in pats:
        if kind == "text":
            if obj in t: return True
        else:
            if obj.search(t): return True
    return False

def _pass_keyword_filter(title: str, *extras: str) -> bool:
    """
    포함 키워드가 비어 있으면 전체 포함.
    제외 키워드는 어떤 하나라도 매치되면 탈락.
    여러 필드를 합쳐 검사(가변 인자).
    """
    cat = " ".join([title] + [e for e in extras if e]).strip()
    if _EXC_PAT and _match_patterns(cat, _EXC_PAT):
        return False
    if _INC_PAT:
        return _match_patterns(cat, _INC_PAT)
    return True


# 경계(토큰) 인식: '대구'는 잡고 '해운대구'는 안 잡음
_HANGUL_ALNUM = r"[0-9A-Za-z가-힣]"
def _contains_token(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    s = _norm_text(text)  # 이미 있으니 재사용
    for p in patterns:
        # 앞뒤가 한글/영문/숫자가 아니면 '단어'로 간주
        if re.search(rf"(?<!{_HANGUL_ALNUM}){re.escape(p)}(?!{_HANGUL_ALNUM})", s):
            return True
    return False


# [ADD] 로그용 사업소 문자열 포맷터
def _fmt_offices_for_log(val):
    if not val:
        return ""
    if isinstance(val, (list, tuple, set)):
        return ", ".join(str(x) for x in val if x)
    # 문자열로 들어온 경우 'A/B' 를 'A, B' 로 보기 좋게
    return str(val).replace("/", ", ")
# === apt_list.csv 캐시/조회 (우선순위 1) ===
import csv


def _get_resource_path(relative_path):
    """
    PyInstaller로 빌드된 환경에서 리소스 경로를 찾는 헬퍼 함수
    """
    # PyInstaller로 빌드된 경우, sys._MEIPASS는 임시 압축 해제 경로를 가리킵니다.
    try:
        base_path = sys._MEIPASS
    # 일반 파이썬 환경인 경우, 현재 스크립트의 경로를 사용합니다.
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)

@lru_cache(maxsize=1)
def _load_apt_list_cache() -> dict:
    """
    apt_list.csv를 PyInstaller 번들 또는 일반 경로에서 안전하게 로드합니다.
    """
    path_to_use = _get_resource_path("apt_list.csv")
    if not os.path.isfile(path_to_use):
        print("[Warning] apt_list.csv 파일을 찾을 수 없어 단지 정보 매핑이 제한됩니다.")
        return {}

    db = {}
    with open(path_to_use, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            code = (row.get("kapt_code") or row.get("kaptCode") or "").strip()
            if not code:
                continue
            db[code] = {
                "address": (row.get("address") or row.get("addr") or "").strip(),
                "office": (row.get("office") or "").strip(),
                "bjd_code": (row.get("bjd_code") or row.get("bjdCode") or "").strip()
            }
    return db

def lookup_apt_by_code(kapt_code: str) -> tuple[str, str, str]:
    """
    (address, office, bjd_code) 세트 반환.
    - apt_list.csv 캐시에서 직접 조회
    - kapt_code가 없거나 매칭 안 되면 ("", "", "") 반환
    """
    if not kapt_code:
        return ("", "", "")

    m = _load_apt_list_cache()
    hit = m.get(kapt_code.strip())
    if not hit:
        return ("", "", "")

    return (
        hit.get("address", "").strip(),
        hit.get("office", "").strip(),
        hit.get("bjd_code", "").strip()
    )



def http_get_json(url: str, params: dict, *, retries: int = 3, timeout: int = 12, backoff: float = 0.8):
    """
    안전 JSON GET:
    - JSON 아닌 응답(빈 문자열/HTML/XML)일 때 None 반환
    - 5xx/429/타임아웃은 지수 백오프로 재시도
    - 응답 Content-Type 검사 및 본문 선행문자 검사
    - BOM/제로폭 문자 안전 파싱
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)

            # 재시도 대상 상태코드
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                raise last_err

            # 204 No Content 등
            if r.status_code == 204 or not r.text:
                return None

            # Content-Type 기본 점검
            ctype = (r.headers.get("Content-Type") or "").lower()
            text = (r.text or "").lstrip()

            # JSON 합리성 점검: content-type 또는 선행 문자
            looks_json = ("json" in ctype) or text.startswith("{") or text.startswith("[")
            if not looks_json:
                # K-APT/나라장터가 가끔 HTML(점검/오류)을 주는 케이스 방지
                return None

            # --- BOM/제로폭 문자 안전 파싱 ---
            try:
                return r.json()
            except Exception:
                try:
                    # BOM 제거 디코딩
                    txt = r.content.decode("utf-8-sig", errors="replace")
                    return json.loads(txt)
                except Exception:
                    # 제어문자 제거 후 파싱
                    txt = (r.text or "")
                    # 흔한 문제문자 제거
                    for bad in ("\ufeff", "\u200b", "\u200c", "\u200d"):
                        txt = txt.replace(bad, "")
                    txt = txt.strip()
                    return json.loads(txt)

        except Exception as e:
            last_err = e
            if attempt < retries:
                # 지수 백오프
                time.sleep(backoff ** attempt)
                continue
            # 마지막 시도 실패
            return None
    # 논리적으로 여기 오지 않지만, 안전망
    return None

def to_ymd(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    # 2025-08-12T10:20:00 → 2025-08-12
    return s.split("T")[0].split()[0]

def _as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, list) and x:
        return x[0]
    return {}



# =================================================================
# [추가 기능] KEA 고효율에너지기자재 인증정보 API 조회
# =================================================================

import xml.etree.ElementTree as ET

# KEA API 엔드포인트
KEA_API_URL = "http://apis.data.go.kr/B553530/CRTIF/CRITF_01_LIST"

def _normalize_model(model: str) -> str:
    """모델명을 API 조회에 적합하게 정규화합니다."""
    if not model:
        return ""
    return model.strip()

@lru_cache(maxsize=4096)
def kea_has_model_cached(model: str) -> bool | None:
    """
    kea_has_model 함수의 결과를 캐시하여 중복 API 호출을 방지합니다.
    """
    return kea_has_model(model)

def kea_has_model(model: str) -> bool | None:
    """
    KEA API에서 모델 존재 여부를 확인합니다.
    - True: 인증 모델 존재
    - False: 조회했지만 없음
    - None: API 오류로 판정 불가
    """
    model_q = _normalize_model(model)
    if not model_q or "없음" in model_q or "필요" in model_q:
        return None

    params = {
        "serviceKey":_cfg("KEA_SERVICE_KEY"), # config.py에 KEA_SERVICE_KEY가 있어야 합니다.
        "pageNo": 1,
        "numOfRows": 10,
        "apiType": "json",
        "q2": model_q  # 모델명 검색 파라미터
    }

    try:
        response = SESSION.get(KEA_API_URL, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        total_count = data.get('totalCount')
        if total_count is None and 'response' in data:
            body = data.get('response', {}).get('body', {})
            if body:
                total_count = body.get('totalCount', 0)
        if isinstance(total_count, str) and total_count.isdigit():
            total_count = int(total_count)
        if isinstance(total_count, int):
            return total_count > 0
    except requests.exceptions.RequestException as e:
        print(f"  [KEA API Error] API 호출 실패 (모델: {model_q}): {e}")
    except (ValueError, KeyError) as e:
        print(f"  [KEA API Error] 응답 데이터 처리 실패 (모델: {model_q}): {e}")
    return None


def kea_check_certification(model: str) -> str:
    """
    실제 인증여부 판정:
    - 인증(O)
    - 미인증(X)
    - 판정불가(확인필요)
    """
    res = kea_has_model_cached(model)
    if res is True:
        return "O"
    elif res is False:
        return "X"
    return "확인필요"

def kea_cert_with_similarity(model: str) -> str:
    """
    KEA API 호출 후, 응답 리스트와 모델명을 유사도 비교.
    유사도 0.75 이상이면 인증 처리.
    """
    model_q = _normalize_model(model)
    if not model_q:
        return "확인필요"

    try:
        params = {
            "serviceKey": _cfg("KEA_SERVICE_KEY"),
            "pageNo": 1,
            "numOfRows": 50,
            "apiType": "json",
            "q2": model_q
        }
        r = SESSION.get(KEA_API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        # KEA JSON 구조 파악
        items = []
        if "items" in data:
            items = data["items"]
        elif "response" in data:
            body = data["response"].get("body", {})
            items = body.get("items", body.get("item", []))

        if not isinstance(items, list):
            items = [items]

        # 유사도 기반 인증 판정
        for it in items:
            kea_model = it.get("mdlpNm") or it.get("modelNm") or ""
            if not kea_model:
                continue

            sim = model_similarity(model, kea_model)
            if sim >= 0.75:   # ★ 유사도 기준 (병훈님 필요 시 조정 가능)
                return "O"   # 인증된 것으로 판단
                
        # API는 응답했으나 유사도가 낮아 인증 아님
        return "X"

    except Exception as e:
        print(f"[KEA Similarity Error] 모델:{model}, err:{e}")
        return "확인필요"


# =========================
# DB
# =========================
Session = sessionmaker(bind=engine)
session = Session()

def _upsert_with_target(n: dict, conflict_cols: List[str]):
    """지정한 conflict 타겟으로 upsert 시도"""
    stmt = sqlite_insert(Notice).values(**n)
    update_columns = {c.name: c for c in stmt.excluded
                      if c.name not in ("id", "detail_link", "model_name")}
    stmt = stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_=update_columns
    )
    session.execute(stmt)

# [수정] DB 저장 로직: source_system을 포함하도록 upsert_notice 수정
def upsert_notice(n: dict):
    """
    - source_system을 포함한 복합키로 upsert 시도
    """
    try:
        # `database.py`에서 UniqueConstraint가 변경되었으므로, 그에 맞는 컬럼 사용
        _upsert_with_target(n, ["source_system", "detail_link", "model_name", "assigned_office"])
    except IntegrityError:
        session.rollback()
        # IntegrityError가 발생하면 로그를 남기거나 다른 처리를 할 수 있습니다.
        print(f"  [Warn] Upsert failed, possibly due to constraint violation: {n.get('detail_link')}")


# =========================
# 필터(관심도)  ← 우선순위 1번 (먼저 거릅니다) — 강화판
# =========================

def _norm_text(*texts: str) -> str:
    """간단 정규화: 소문자, 공백 축약, 괄호/특수문자 최소 제거"""
    
    s = " ".join((t or "") for t in texts).lower()
    s = re.sub(r"[\(\)\[\]{}<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 표기/동의어 포함 (소문자 비교 기준)
DEVICE_KEYWORDS = [
    # 조명/LED
    "led", "엘이디", "발광다이오드", "조명", "가로등", "보안등", "터널등", "스마트 LED", "스마트LED",
    # 회전/동력
    "모터", "전동기", "펌프", "블로워", "팬", "에어드라이어", "pcm",
    # 기타
    "히트펌프", "냉동기", "터보압축기", "김건조기",
    # 전력/제어
    "변압기", "트랜스", "인버터", "인버터 제어형",
    # 설비/기계
    "공기압축기", "사출성형기",
    # 수송/승강
    "승강기", "엘리베이터"
]

IMPROVEMENT_KEYWORDS = [
    "보수", "개선", "성능개선", "효율개선", "개체", "교체",
    "정비", "개량", "리모델링", "개보수", "노후교체", "업그레이드",
]

ENERGY_PROGRAM_KEYWORDS = [
    "고효율", "에너지절감", "효율향상", "에너지절약", "전력기금",
    "지원사업", "보조금", "정부지원", "효율등급", "에너지이용합리화"
]

# 무관/제외 (강한 제외)
HARD_DENY_KEYWORDS = [
    "인력", "파견", "용역", "교육용역", "컨설팅", "위탁운영", "임차", "위탁교육", "교육훈련",
    "급식", "인쇄", "소프트웨어", "유지보수", "토목", "건축", "조경", "도로", "클라우드", "비품", "사무용품",
    "사무가구", "비품구매", "문구류", "의료소모품", "식자재", "세탁물", "청소용역", "해운대구", "전광판", "저소득층",
    "신축", "횡단보도", "태양광", "벽시계", "모니터", "무드등", "연필꽂이", "교통신호기", "해운대", "OA기기", "취약계층", 
    # ▼▼▼ [수정] 대구본부 관할 외 경북 지역을 여기에 추가하여 즉시 제외 ▼▼▼
    "안동", "상주", "문경", "의성", "예천", "영주",
    "봉화", "청송", "영양", "울진", "구미", "군위", "울릉", "기관명 없음"
]

# 권역 판별(텍스트에 다른 광역권만 분명히 나오면 컷)
TARGET_REGION_KEYWORDS = [
    "대구", "대구광역시", "경북", "경상북도",
    "포항", "경주", "경산", "김천", "영천", "칠곡", "성주", "청도", "고령", "영덕"
]
OTHER_REGION_KEYWORDS = [
    # 수도권
    "서울", "강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천",
    "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동",
    "성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중랑",
    "경기", "수원", "성남", "고양", "용인", "부천", "안산", "안양",
    "남양주", "화성", "평택", "의정부", "시흥", "파주", "김포", "광명",
    "광주", "군포", "이천시", "오산", "안성", "하남", "의왕", "양주", 
    "포천", "여주", "양평", "가평", "연천",
    "인천", "계양", "남동", "미추홀", "부평", "연수", "강화", "옹진", "경기도",

    # 강원권
    "강원", "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척", "강원도",
    "홍천", "횡성", "영월", "평창", "정선", "철원", "화천", "양구",
    "인제", "고성", "양양",

    # 충청권
    "충북", "청주", "충주", "제천", "보은", "옥천", "영동", "충청북도", "충청남도", 
    "괴산", "음성", "단양", "충남",
    "충남", "천안", "공주", "보령", "아산", "서산", "논산", "계룡",
    "당진", "금산", "부여", "서천", "청양", "홍성", "예산", "태안",
    "대전", "유성", "대덕", "세종",

    # 전라권
    "전북", "전주", "군산", "익산", "정읍", "남원", "김제", "전라북도", "전라남도",
    "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안",
    "전남", "목포", "여수", "순천", "나주", "광양",
    "담양", "곡성", "구례", "고흥", "보성", "화순", "장흥", "강진",
    "해남", "영암", "무안", "함평", "영광", "장성", "완도", "진도", "신안",
    "광주", "광산",

    # 경상권 (대구본부 제외)
    "부산", "영도", "부산진", "동래", "해운대", "사하", "금정", "경상남도",
    "연제", "사상", "기장",
    "울산", "울주",
    "경남", "창원", "김해", "양산", "진주", "거제", "통영",
    "사천", "밀양", "함안", "창녕", "고성", "남해", "하동",
    "산청", "함양", "거창", "합천",


    # 제주권
    "제주", "서귀포",
]


def is_relevant_text(*texts: str) -> bool:
    """
    강화된 관심 공고 필터:
      1) 군위/하드거절 단어 있으면 즉시 제외
      2) 타 권역 키워드가 명시돼 있고, 타깃 권역이 없으면 제외
      3) 스코어 >= 2만 통과 (장비=2 / 에너지=1 / 개선=1)
         - 장비 키워드 1개만 있어도 통과 (2점)
         - 개선(1)+에너지(1) 조합도 통과 (2점)
    """
    s = _norm_text(*texts)

    # 1) 최우선 제외
    if any(k in s for k in (kw.lower() for kw in HARD_DENY_KEYWORDS)):
        return False

    # 2) 지역 오탐 컷 (다른 권역 + 우리 권역 부재)
    has_other = any(k in s for k in (kw.lower() for kw in OTHER_REGION_KEYWORDS))
    has_target = any(k in s for k in (kw.lower() for kw in TARGET_REGION_KEYWORDS))
    if has_other and not has_target:
        return False

    # 3) 가중치 스코어
    score = 0
    if any(k in s for k in (kw.lower() for kw in DEVICE_KEYWORDS)):
        score += 2
    if any(k in s for k in (kw.lower() for kw in ENERGY_PROGRAM_KEYWORDS)):
        score += 1
    if any(k in s for k in (kw.lower() for kw in IMPROVEMENT_KEYWORDS)):
        score += 1

    return score >= 2

def _safe_hint_match(text: str, hint_key: str) -> bool:
    """
    CLIENT_HINTS 키워드가 텍스트에 있을 때, 불필요한 전국 오탐을 줄이기 위한 가드.
    - '군위' 포함 시 무조건 불가
    - 너무 일반적인 키워드는 '대구/경북/포항' 중 하나의 맥락 또는
      지명(구/군/시/읍/면/동) 동반 시만 허용하도록 확장 가능.
    """
    s = _norm_text(text)
    if "군위" in s:
        return False

    # 예) '중구청', '북구청' 같은 전국 일반어는 CLIENT_HINTS에 넣지 않았고,
    #     넣더라도 아래처럼 '대구 ' 접두가 없는 경우 컷 가능 (필요 시 확장)
    general_local_terms = ["중구청", "북구청", "남구청", "서구청"]
    if any(g in hint_key for g in general_local_terms):
        # CLIENT_HINTS에는 '대구 중구청'처럼 넣어두었으므로 여긴 사실상 패스
        return False

    # '대구/경북/포항' 맥락 체크 (과도 컷 방지 위해 완전 하드 조건은 아님. 필요시 강화)
    has_context = any(k in s for k in ["대구", "경북", "경상북도", "포항", "경주", "경산", "김천", "영천", "칠곡", "성주", "청도", "고령", "영덕"])
    # 너무 일반적인 기업/기관명 키워드가 생길 경우, 맥락 없으면 컷하도록 추가
    # if hint_key in ["환경공단", "시설공단", "본사", "본부"] and not has_context:
    #     return False

    return True


# =========================
# 특수권역/지사 후보 정의
# =========================
DAEGU_OFFICES      = ["직할", "동대구지사", "서대구지사", "남대구지사"]  # 대구권 전체 지사
GYONGBUK_OFFICES   = ["포항지사", "북포항지사", "경주지사", "경산지사", "김천지사", "영천지사", "칠곡지사", "성주지사", "청도지사", "고령지사", "영덕지사"]

# 달서구/달성군/포항시 북구 = 특수권역 (구까지만 나오면 복수 후보)
SPECIAL_GU_PATTERNS = [
    # (정규식, 후보 지사 2개, '전체'에서 하나만 대표로 쓰는 기본지사)
    (re.compile(r"(대구광역시\s*)?달서구"), ["남대구지사", "서대구지사"], "남대구지사"),
    (re.compile(r"(대구광역시\s*)?달성군"), ["남대구지사", "동대구지사"], "남대구지사"),
    (re.compile(r"(경상북도\s*)?포항시\s*북구"), ["포항지사", "북포항지사"], "포항지사"),
]

# 동/읍/면/로/길 레벨 식별
def has_dong_level(addr: str) -> bool:
    if not addr:
        return False
    return any(t in addr for t in ("동", "읍", "면", "로", "길"))

def _decorate_candidates_in_addr(addr: str, a: str, b: str) -> str:
    """주소 표시는 '원주소\n(A/B)' 형태로"""
    addr = (addr or "").strip()
    if not addr:
        return f"관할지사 확인 필요\n({a}/{b})"
    return f"{addr}\n({a}/{b})"

def _special_gu_offices_if_match(addr: str) -> Optional[List[str]]:
    for pat, candidates, _default in SPECIAL_GU_PATTERNS:
        if pat.search(addr):
            return candidates
    return None

def _assign_office_by_addr(addr: str) -> Optional[str]:
    """assign_offices_by_address 결과를 단일 지사로 축약해 반환"""
    offices = assign_offices_by_address(addr)
    if len(offices) == 1:
        return offices[0]
    return None

def decorate_address_with_candidates(addr: str, offices: List[str]) -> str:
    if len(offices) >= 2:
        return _decorate_candidates_in_addr(addr, offices[0], offices[1])
    return addr or ""

import re
from typing import List

def assign_offices_by_address(addr: str) -> List[str]:
    """
    주소만으로 지사 후보를 결정.
    - 동/읍/면까지 나오면 단일 지사 세분화
    - 달서구/달성군/포항시 북구는 '구'까지만 나오면 2 후보 반환 가능
    - 포항(북구 제외)은 포항지사
    - 매칭 불가 시 [] 반환(상위 단계에서 힌트/후속 처리)
    """
    if not addr:
        return []

    s = re.sub(r"\s+", "", addr)  # 공백 제거본

    # ── 1) 대구 권역(우선) ──
    if "대구광역시" in s or s.startswith("대구"):
        if has_dong_level(addr):
            if re.search(r"대구광역시(중구|북구)", s):
                return ["직할"]
            if re.search(r"대구광역시(동구|수성구)", s):
                return ["동대구지사"]
            if re.search(r"대구광역시(서구|남구)", s):
                return ["서대구지사"]
            if "대구광역시달서구" in s:
                if re.search(r"(감삼동|두류동|본리동|성당동|죽전동)", addr):
                    return ["서대구지사"]
                return ["남대구지사"]
            if "대구광역시달성군" in s:
                if re.search(r"(다사읍|하빈면)", addr):
                    return ["서대구지사"]
                if "가창면" in s:
                    return ["동대구지사"]
                return ["남대구지사"]
            # 대구는 맞지만 구/군 판독 불가 → 상위 처리
        else:
            if re.search(r"대구광역시(중구|북구)", s):
                return ["직할"]
            if re.search(r"대구광역시(동구|수성구)", s):
                return ["동대구지사"]
            if re.search(r"대구광역시(서구|남구)", s):
                return ["서대구지사"]
            if "대구광역시달서구" in s or "대구광역시달성군" in s:
                # 필요 시 SPECIAL_GU_PATTERNS로 후보 2개 처리
                return []

    # ── 2) 포항 권역 ──
    if "포항시" in s or s.startswith("포항"):
        if "포항시북구" in s:
            if has_dong_level(addr):
                if re.search(r"(흥해|송라|신광|청하|기계|기북|죽장)", addr):
                    return ["북포항지사"]
                return ["포항지사"]
            return ["포항지사", "북포항지사"]
        return ["포항지사"]

    # ── 3) 특수 패턴(구까지만 등) ──
    for pat, candidates, _default in SPECIAL_GU_PATTERNS:
        if pat.search(addr):
            return candidates

    # ── 4) 경북권(그 외 시군) ──
    if "경주시" in addr:  return ["경주지사"]
    if "경산시" in addr:  return ["경산지사"]
    if "김천시" in addr:  return ["김천지사"]
    if "영천시" in addr:  return ["영천지사"]
    if "칠곡군" in addr:  return ["칠곡지사"]
    if "성주군" in addr:  return ["성주지사"]
    if "청도군" in addr:  return ["청도지사"]
    if "고령군" in addr:  return ["고령지사"]
    if "영덕군" in addr:  return ["영덕지사"]

    # ── 5) 기타 ──
    if any(g in addr for g in ["동구", "수성구"]):
        return ["동대구지사"]
    if any(g in addr for g in ["서구", "남구"]):
        return ["서대구지사"]
    if "달서구" in addr or "달성군" in addr:
        return []

    return []

# =========================
# UsrInfo(상세주소) & Mall(시군구) 우선순위 선택
# =========================
def get_full_address_from_usr_info(dminstt_code: str) -> Optional[str]:
    """
    UsrInfoService.getDminsttInfo (코드 기준)
    - inqryDiv=2(변경일 기준) + 12개월 기간 필수
    - adrs + dtlAdrs → 상세주소, 없으면 rgnNm fallback
    - 내부 스로틀(120ms) + 단순 캐시 적용
    """
    if not dminstt_code:
        return None

    # --- 간단 캐시 & 스로틀(함수 속성 사용, 외부 코드 수정 불필요) ---
    import time
    from datetime import datetime, timedelta

    if not hasattr(get_full_address_from_usr_info, "_cache"):
        get_full_address_from_usr_info._cache = {}  # type: ignore[attr-defined]
    if not hasattr(get_full_address_from_usr_info, "_last_call"):
        get_full_address_from_usr_info._last_call = 0.0  # type: ignore[attr-defined]

    _cache: dict = get_full_address_from_usr_info._cache  # type: ignore[attr-defined]
    _last_call: float = get_full_address_from_usr_info._last_call  # type: ignore[attr-defined]

    # 캐시 조회(프로세스 생존 동안 유지)
    if dminstt_code in _cache:
        return _cache[dminstt_code]

    # 스로틀: 최소 120ms 간격
    wait = 0.12 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)

    # 12개월 기간(문서 제한 고려)
    end = datetime.now()
    start = end - timedelta(days=365)
    inqryBgnDt = start.strftime("%Y%m%d") + "0000"
    inqryEndDt = end.strftime("%Y%m%d") + "2359"

    params = {
        "ServiceKey": _cfg("NARA_SERVICE_KEY"),
        "type": "json",
        "inqryDiv": "2",                 # ✅ 변경: 기간(변경일) 기준
        "inqryBgnDt": inqryBgnDt,        # ✅ 12개월 범위 시작
        "inqryEndDt": inqryEndDt,        # ✅ 12개월 범위 끝
        "dminsttCd": dminstt_code,       # 코드 기준 조회
        "numOfRows": "1",
        "pageNo": "1",
    }

    try:
        data = http_get_json(api_url(USR_INFO_PATH), params)
        body = _as_dict(data.get("response", {}).get("body"))
        items = _as_items_list(body)
        if items:
            it = items[0]
            full = f"{it.get('adrs','')}".strip()
            dtl  = f"{it.get('dtlAdrs','')}".strip()
            text = (full + " " + dtl).strip() or it.get("rgnNm")
            # 캐시에 적재
            _cache[dminstt_code] = text
            return text
    except Exception as e:
        print(f"  [Warn] 사용자정보 API 실패: {dminstt_code} ({e})")
    finally:
        # 마지막 호출시각 갱신
        get_full_address_from_usr_info._last_call = time.time()  # type: ignore[attr-defined]

    return None


def parse_dminstt_code_from_complex(s: str) -> Tuple[Optional[str], Optional[str]]:
    """'[코드^이름^기관명]|[...]' 형식에서 첫 항목의 코드/명 파싱"""
    if not s:
        return None, None
    try:
        first = s.strip("[]").split("],[")[0].strip("[]")
        parts = first.split("^")
        if len(parts) >= 3:
            return parts[1], parts[2]
    except Exception:
        pass
    return None, None
    
def guess_mall_addr(item: dict) -> Optional[str]:
    keys_try = [
        # 공통/납품요구
        "insttAddr","dmndInsttAddr","dminsttAddr","adres","adrs","addr","adresCn",
        "lc","instNmAddr","insttAdres","dminsttAdres","dmndInsttAdres",
        "insttZipAddr","zipAdres",
        "dlvrReqInsttAddr","dlvrReqAddr","dlvrAddr","destAddr","delivAddr",

        # ✅ 계약완료(물품)에서 자주 보이는 키
        "cntrctInsttAddr","cntrctInsttAdres","cntrctInsttZipAddr",
        "cntrctInsttRgnNm",
        "prchseInsttAddr","prchseInsttAdres","prchseInsttZipAddr",
        "prchseInsttRgnNm",
        "insttRgnNm","dminsttRgnNm",  # 지역명만 내려오는 경우도 표시
    ]
    for k in keys_try:
        v = item.get(k)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip()
    return None


def _pick_addr_by_priority(client_code: Optional[str], mall_addr: Optional[str]) -> Tuple[str, str]:
    """
    반환: (선택주소, source)  # source: 'usr' | 'mall' | 'none'
    """
    usr_addr = get_full_address_from_usr_info(client_code) if client_code else None
    if usr_addr:
        return usr_addr.strip(), "usr"
    if mall_addr:
        return str(mall_addr).strip(), "mall"
    return "", "none"

# =========================
# 기관명 힌트 → 저장(관할불명) 판단
# (중복/전국 오탐 방지를 위해 구체화: '대구 동구청' 등 네임스페이스 부여)
# 달서구청 키워드는 제거(특수권역은 주소 로직으로 처리)
# =========================
CLIENT_HINTS = {
    # ========================
    # 대구광역시 직할 (중구/북구, 본부급)
    # ========================
    "대구광역시청": "직할", "대구시청": "직할", "대구광역시 본청": "직할",
    "대구 중구청": "직할", "대구광역시 중구": "직할",  # <-- ADD
    "대구 북구청": "직할", "대구광역시 북구": "직할",  # <-- ADD
    "대구 중구청": "직할", "대구 북구청": "직할",
    "한국전력공사 대구본부": "직할", "한전 대구본부": "직할",
    "대구환경공단 본부": "직할", "대구시설공단 본사": "직할",
    "엑스코": "직할", "영진전문대": "직할", "경북대학교": "직할", "경북대병원": "직할",
    "대구지방경찰청": "직할", "대구북부경찰서": "직할", "경북대학교 공과대학": "직할",
    "대구시설공단 북구사업소": "직할", "대구농수산물유통관리공사": "직할",
    "대구도남": "직할", "대구역한라하우젠트센트로": "직할",
    # ========================
    # 동대구지사 (동구, 수성구, 가창면)
    # ========================
    "대구 동구청": "동대구지사", "대구 수성구청": "동대구지사", "수성구": "동대구지사",
    "대구경북첨단의료산업진흥재단": "동대구지사", "첨복재단": "동대구지사",
    "국립대구박물관": "동대구지사", "한국가스공사 본사": "동대구지사", 
    "대구미술관": "동대구지사", "수성대학교": "동대구지사",
    "동대구역": "동대구지사", "달성군 가창면": "동대구지사",
    "대구환경공단 동부사업소": "동대구지사", "대구동부경찰서": "동대구지사", "대구수성경찰서": "동대구지사",
    "대구 동구청": "동대구지사", "대구광역시 동구": "동대구지사", # <-- ADD
    "대구 수성구청": "동대구지사", "대구광역시 수성구": "동대구지사", # <-- ADD
    "대구창의융합교육원": "동대구지사", "대구율하": "동대구지사", 

    # ========================
    # 서대구지사 (서구, 남구, 달서구/달성군 일부)
    # ========================
    "대구 서구청": "서대구지사", "대구 남구청": "서대구지사",
    "두류공원": "서대구지사", "대구의료원": "서대구지사",
    "계명대학교 동산병원": "서대구지사", "대구가톨릭대학교병원": "서대구지사",
    "서대구산업단지": "서대구지사", "서대구산단": "서대구지사",
    "대구환경공단 서부사업소": "서대구지사", "대구시설공단 남구사업소": "서대구지사",
    "대구서부경찰서": "서대구지사", "대구남부경찰서": "서대구지사",
    "대구 서구청": "서대구지사", "대구광역시 서구": "서대구지사", 
    "대구광역시립서부도서관": "서대구지사", "죽곡정수사업소": "서대구지사",
    "대구광역시 상수도사업본부 죽곡정수사업소": "서대구지사",

    # ========================
    # 남대구지사 (달서구/달성군 대부분)
    # ========================
    "달성군청": "남대구지사/서대구지사/동대구지사",
    "달서구청": "남대구지사/서대구지사",
    "대구광역시 달서구": "남대구지사/서대구지사", 
    "대구광역시 달성군": "남대구지사/서대구지사/동대구지사",
    "성서산업단지": "남대구지사", "성서산단": "남대구지사",
    "대구국가산업단지": "남대구지사", "대구국가산단": "남대구지사",
    "국립대구과학관": "남대구지사", "테크노폴리스": "남대구지사",  "계명대학교": "서대구지사",
    "대구환경공단 남부사업소": "남대구지사", "대구시설공단 달서사업소": "남대구지사",
    "대구기계부품연구원": "남대구지사", "지역난방공사 대구": "남대구지사", "대광텍스타일": "남대구지사",
    "월배국민체육센터": "남대구지사", "유가 테크노폴리스": "남대구지사",
    "달성종합스포츠파크": "남대구지사", "대구경북과학기술원": "남대구지사",
    "대구학생문화센터": "남대구지사", "달성중": "남대구지사",

    # ========================
    # 포항지사 (포항 남구)
    # ========================
    "포항시청": "포항지사", "포항 남구청": "포항지사", "포스코": "포항지사",
    "포항공과대학교": "포항지사", "포항시립미술관": "포항지사",
    "포항환경관리원": "포항지사", "포항남부경찰서": "포항지사",
    "포항의료원": "포항지사", "포항블루밸리국가산단": "포항지사",
    "포항지방해양수산청": "포항지사", "해양수산부 포항지방해양수산청": "포항지사",
    
    # ========================
    # 북포항지사 (포항 북구)
    # ========================
    "포항 북구청": "북포항지사", "포항테크노파크": "북포항지사", "한동대학교": "북포항지사",
    "포항북부경찰서": "북포항지사", "포항융합기술산단": "북포항지사",
    "포항국토관리사무소": "북포항지사", "국토교통부 부산지방국토관리청 포항국토관리사무소": "북포항지사",
    "한국농어촌공사 경북지역본부 포항지사": "북포항지사", "포항흥해": "북포항지사",
    "경상북도 환동해지역본부": "북포항지사", "환동해지역본부": "북포항지사", "경상북도 동부청사": "북포항지사",
    # ========================
    # 경주지사
    # ========================
    "경주시청": "경주지사", "경주화백컨벤션센터": "경주지사", "경주엑스포": "경주지사",
    "동국대학교 경주캠퍼스": "경주지사", "한국수력원자력 본사": "경주지사", "보문단지": "경주지사", 
    "경주의료원": "경주지사", "경주외동산단": "경주지사", "경주경찰서": "경주지사", "엑스포": "경주지사",
    "한국수력원자력": "경주지사", "APEC": "경주지사",
   

    # ========================
    # 경산지사
    # ========================
    "경산시청": "경산지사", "영남대학교": "경산지사", "대구가톨릭대학교": "경산지사",
    "경산지식산업지구": "경산지사", "경산경찰서": "경산지사", "대구한의대학교": "경산지사",
    "호산대학교": "경산지사", "대구대학교": "경산지사",
   
    # ========================
    # 김천지사
    # ========================
    "김천시청": "김천지사", "한국도로공사 본사": "김천지사", "혁신도시(김천)": "김천지사",
    "김천의료원": "김천지사", "김천경찰서": "김천지사",
   

    # ========================
    # 영천지사
    # ========================
    "영천시청": "영천지사", "영천하이테크파크": "영천지사", "영천경찰서": "영천지사",

    # ========================
    # 교육기관 (Education Offices)
    # ========================
    "대구광역시달성교육지원청": "남대구지사/서대구지사/동대구지사",
    "대구광역시달성교육청": "남대구지사/서대구지사/동대구지사",
    "대구광역시남부교육지원청": "서대구지사/남대구지사",
    "대구광역시남부교육청": "서대구지사/남대구지사",
    "대구광역시서부교육지원청": "서대구지사/직할",
    "대구광역시서부교육청": "서대구지사/직할",
    "대구광역시동부교육지원청": "동대구지사/직할",
    "대구광역시동부교육청": "동대구지사/직할",
}
try:
    from client_hints_schools import CLIENT_HINTS_SCHOOLS
    CLIENT_HINTS.update(CLIENT_HINTS_SCHOOLS)
except Exception as e:
    print("[school hints] load failed:", e)


BROAD_KEYWORD_OFFICE_MAP = {
    "경주시": "경주지사", "경주": "경주지사",
    "경산시": "경산지사", "경산": "경산지사",
    "김천시": "김천지사", "김천": "김천지사",
    "영천시": "영천지사", "영천": "영천지사",
    "칠곡군": "칠곡지사", "칠곡": "칠곡지사",
    "성주군": "성주지사", "성주": "성주지사",
    "청도군": "청도지사", "청도": "청도지사",
    "고령군": "고령지사", "고령": "고령지사",
    "영덕군": "영덕지사", "영덕": "영덕지사",
}



def assign_offices_by_keywords(client_name: str, project_name: str) -> List[str]:
    """주소로 못 정하면, 수요기관명 + 사업명(제목)에서 힌트 추론"""
    text = f"{client_name or ''} {project_name or ''}"
    # '군위' 우선 제외(deny가 걸러주지만, 여기서도 1차 방어)
    if "군위" in text:
        return []
    # 구체 키워드 우선 매칭
    # 1순위: 가장 구체적인 전체 기관명으로 검색 (예: "대구 동구청")
    # sorted를 통해 긴 이름(더 구체적인 이름)을 먼저 비교
    for k, office in sorted(CLIENT_HINTS.items(), key=lambda x: len(x[0]), reverse=True):
        if k and k in text:
            # office가 "A/B" 형태일 수 있으므로 split 후 리스트로 반환
            return office.split('/')

    # 2순위: 관할 시/군 키워드로 단일 사업소 검색 (예: "성주", "경주")
    # _contains_token을 사용하여 '성주산' 같은 단어의 일부가 일치하는 오류 방지
    for keyword, office in BROAD_KEYWORD_OFFICE_MAP.items():
        if _contains_token(text, [keyword]):
            return [office]
            
    # 대구/경북 대역 키워드로 '관할불명(분배) 후보' 반환
    if any(t in text for t in ["대구광역시", " 대구", "대구 ", "대구"]):
        return DAEGU_OFFICES  # 대구권 전 지사에 노출

    return []


def _assign_office_by_client_name(client_name: str) -> Optional[str]:
    """CLIENT_HINTS를 기반으로 기관명에서 직접 관할 지사를 찾습니다."""
    if not client_name:
        return None
    for kw in sorted(CLIENT_HINTS.keys(), key=len, reverse=True):
        if kw in client_name:
            return CLIENT_HINTS[kw]
    return None



def filter_duplicates(items):
    """source_system, detail_link, model_name, assigned_office 중복 필터"""
    seen = set()
    unique = []
    for it in items:
        key = (it["source_system"], it["detail_link"], it["model_name"], it["assigned_office"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique


# =========================
# 저장 로직
# =========================
def _save_dual_office_rows(base_notice: dict, addr: str, offices: List[str]):
    """
    A/B 특수권역은 이제 단일 행으로 저장합니다.
    - assigned_office: "A/B" (GUI에서 줄바꿈 처리)
    - address: API 원본 주소만 저장 (장식/주석 금지)
    - status: 원본 유지 (보조행 저장 없음)
    """
    n = dict(base_notice)
    n["assigned_office"] = f"{offices[0]}/{offices[1]}"
    n["address"] = addr or ""
    n["status"] = n.get("status", "")
    try:
        upsert_notice(n)
        session.commit()
        print(f"  [✅ 저장 완료 (A/B-단일행)] {n.get('assigned_office')} / {n.get('client')}")
    except Exception as e:
        session.rollback()
        print(f"  [Error] A/B 단일행 저장 실패: {e}")


import unicodedata

def _is_exact_lh_dgrb(name: Optional[str]) -> bool:
    t = unicodedata.normalize("NFKC", (name or "").strip())
    return t == "한국토지주택공사 대구경북지역본부"

def expand_and_store_with_priority(
    base_notice: dict,
    client_code: Optional[str],
    mall_addr: Optional[str],
    client_name: Optional[str],
    save: bool = True
):
    
    def _fill_kea_if_needed(n: dict):
        # 타지역 컷을 모두 통과한 후에만 호출됨
        if not USE_KEA_CHECK:
            return
        if n.get("is_certified") != "확인필요":
            return
        m = (n.get("model_name") or "").strip()
    # ✅ KEA 조회 스킵 대상(무의미 모델명)
        SKIP_MODELS = {
            "모델명 없음", "세부내역 미확인", "N/A",
            "계획 단계 확인", "공고 확인 필요", "입찰 확인 필요", "계약 확인 필요",
        }
        if not m or m in SKIP_MODELS:
            return

        try:
            r = kea_has_model_cached(m)
            n["is_certified"] = "O(인증)" if r is True else ("X(미인증)" if r is False else "확인필요")
        except Exception as e:
            # KEA 에러는 저장을 막지 않고, 상태만 '확인필요' 유지
            print(f"  [KEA] 조회 스킵/오류: {e}")

    def _save(n):
        _fill_kea_if_needed(n) 
        if save:
            upsert_notice(n); session.commit()
            print(f"  [✅ 저장 완료] {n.get('assigned_office','')} / {n.get('client')}")
            return None
        else:
            print(f"  [🧺 저장 대기] {n.get('assigned_office','')} / {n.get('client')}")
            return n
    # 타권역만 명시 & 목표권역 부재 시 컷 (기관명까지 포함해 재확인)
    _alltxt_norm = _norm_text(base_notice.get("project_name",""), client_name or "", mall_addr or "")
   
    # [FIX] Add a hard-deny check on the combined text at the very beginning.
    
    if any(k in _alltxt_norm for k in (kw.lower() for kw in HARD_DENY_KEYWORDS)):
        print_exclude_once(base_notice, client_name, mall_addr or "")
        return

    # The existing region check can remain as a secondary filter
    if any(k in _alltxt_norm for k in (kw.lower() for kw in OTHER_REGION_KEYWORDS)) and not any(k in _alltxt_norm for k in (kw.lower() for kw in TARGET_REGION_KEYWORDS)):
        print_exclude_once(base_notice, client_name, mall_addr or "")
        return
    

    # ─────────────────────────────────────────────────────────────
    # [특수] 기관명이 '한국토지주택공사 대구경북지역본부'일 때:
    #  - 학교는 보지 않음
    #  - 우선순위: (1) 사업명 → (2) 기관명 → (3) 주소(마지막)
    if _is_exact_lh_dgrb(client_name):
        project_title = (base_notice.get("project_name") or "").strip()

        # (1) 사업명(제목) 우선
        if project_title:
            offices = assign_offices_by_keywords("", project_title)  # 제목만 전달
            if offices:
                n = dict(base_notice)
                n["assigned_office"] = "/".join(offices) if isinstance(offices, (list, tuple)) else str(offices)
                n["address"] = mall_addr or ""
                return _save(n)

        # (2) 기관명 기반 (단일 매핑 → 키워드)
        if client_name:
            office = _assign_office_by_client_name(client_name)
            if office:
                n = dict(base_notice)
                n["assigned_office"] = office
                n["address"] = mall_addr or ""
                return _save(n)

            offices = assign_offices_by_keywords(client_name, "")
            if offices:
                n = dict(base_notice)
                n["assigned_office"] = "/".join(offices) if isinstance(offices, (list, tuple)) else str(offices)
                n["address"] = mall_addr or ""
                return _save(n)

        # (3) 주소 기반 (마지막에만 시도)
        addr, src = _pick_addr_by_priority(client_code, mall_addr)

        # 필요 시 기관명→UsrInfo 보조(주소가 없거나, usr가 아니고 동레벨 미포함인 경우)
        if USE_NAME_BASED_USRINFO and (not addr or (src != 'usr' and not has_dong_level(addr))) and client_name:
            try:
                name_addr = _usr_addr_by_name_cached(client_name.strip())
                if name_addr:
                    addr, src = name_addr, 'usr'
            except Exception:
                pass

        if addr:
            # 목표 권역 체크(특수 케이스에도 동일 기준 적용)
            if not _is_address_in_scope(addr):
                print_exclude_once(base_notice, client_name, addr)
                return

            # 상세동 없음 + 특수 구 매칭 시 A/B 단일행 저장
            if not has_dong_level(addr):
                offices = _special_gu_offices_if_match(addr)
                if offices and len(offices) == 2:
                    n = dict(base_notice)
                    n["assigned_office"] = f"{offices[0]}/{offices[1]}"
                    n["address"] = addr or ""
                    return _save(n)

            # 단일 오피스 지정
            office = _assign_office_by_addr(addr)
            if office:
                n = dict(base_notice)
                n["assigned_office"] = office
                n["address"] = addr
                return _save(n)

        # 여기까지도 못 잡으면 최종 제외 (주소는 더 안 봄)
        print_exclude_once(base_notice, client_name, mall_addr or "")
        return    
    # === (추가) 기관명 기반 UsrInfo 주소 조회 보조기 (이 함수 내부 캐시/스로틀)
    def _usr_addr_by_name_cached(name: str) -> Optional[str]:
        if not name:
            return None

        # 간단 캐시/스로틀(함수 속성 사용)
        import time
        from datetime import datetime, timedelta

        if not hasattr(_usr_addr_by_name_cached, "_cache"):
            _usr_addr_by_name_cached._cache = {}  # type: ignore[attr-defined]
        if not hasattr(_usr_addr_by_name_cached, "_last_call"):
            _usr_addr_by_name_cached._last_call = 0.0  # type: ignore[attr-defined]

        _cache: dict = _usr_addr_by_name_cached._cache  # type: ignore[attr-defined]
        _last_call: float = _usr_addr_by_name_cached._last_call  # type: ignore[attr-defined]

        if name in _cache:
            return _cache[name]

        wait = 0.12 - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)

        # 최근 12개월 범위 (UsrInfo는 기간 필수)
        end = datetime.now()
        start = end - timedelta(days=365)
        inqryBgnDt = start.strftime("%Y%m%d") + "0000"
        inqryEndDt = end.strftime("%Y%m%d") + "2359"

        params = {
            "ServiceKey": _cfg("NARA_SERVICE_KEY"),
            "type": "json",
            "inqryDiv": "2",                 # 변경일 기준
            "inqryBgnDt": inqryBgnDt,
            "inqryEndDt": inqryEndDt,
            "dminsttNm": name,               # 기관명 기준 조회
            "numOfRows": "1",
            "pageNo": "1",
        }
        try:
            data = http_get_json(api_url(USR_INFO_PATH), params)
            body = _as_dict(data.get("response", {}).get("body"))
            items = _as_items_list(body)
            if not items:
                return None
            it = items[0]
            full = f"{it.get('adrs','')}".strip()
            dtl  = f"{it.get('dtlAdrs','')}".strip()
            text = (full + " " + dtl).strip() or it.get("rgnNm")
            _cache[name] = text
            return text
        except Exception as e:
            print(f"  [Warn] UsrInfo(name) 실패: {name} ({e})")
            return None
        finally:
            _usr_addr_by_name_cached._last_call = time.time()  # type: ignore[attr-defined]

    # 1/2순위: 주소 정보로 배정
    addr, src = _pick_addr_by_priority(client_code, mall_addr)
    # 수정 코드
    if USE_NAME_BASED_USRINFO and (not addr or src != 'usr' and not has_dong_level(addr)) and client_name:
        name_addr = _usr_addr_by_name_cached(client_name.strip())
        if name_addr:
            addr = name_addr
            src = 'usr'



    def _is_address_in_scope(a: str) -> bool:
        s = (a or "").replace(" ", "")
        # 1) 대구 전역 허용
        if "대구" in s:
            return True
        # 2) 경상북도 내 화이트리스트(대상 권역만 허용)
        gb_allow = ["포항시", "경주시", "영덕군", "영천시", "청도군", "경산시", "김천시", "성주군", "칠곡군", "고령군"]
        if "경상북도" in s:
            return any(w.replace(" ", "") in s for w in gb_allow)
        # 그 외 광역은 기본 제외
        return False

    if addr and not _is_address_in_scope(addr):
        print_exclude_once(base_notice, client_name, addr)
        return


    # 1순위: 학교명 우선(나라장터 전용) - 교육청 등 상위기관이 섞여도 '학교'만 보고 단일 지사 배정
    school_office = _assign_office_by_school_name(client_name or "", base_notice.get("project_name","") or "")
    if school_office:
        n = dict(base_notice)
        n["assigned_office"] = school_office      # ✅ 단일 지사만
        n["address"] = mall_addr or ""            # 주소는 mall_addr 유지
        return _save(n)  

    # 2순위:
    if addr:
        # 특수권역: 구까지만 나온 경우 A/B 저장
        if not has_dong_level(addr):
            offices = _special_gu_offices_if_match(addr)
            if offices and len(offices) == 2:
                n = dict(base_notice); n["assigned_office"] = f"{offices[0]}/{offices[1]}"; n["address"] = addr or ""
                #print(f"  [✅ 저장 후보 (A/B-단일행)] {n.get('assigned_office')} / {n.get('client')}")
                return _save(n)
        # 단일 오피스 지정
        office = _assign_office_by_addr(addr)
        if office:
            n = dict(base_notice); n["assigned_office"] = office; n["address"] = addr
            #print(f"  [✅ 저장 후보 ({'상세주소' if src=='usr' else '수요기관주소'})] {n.get('assigned_office')} / {n.get('client')}")
            return _save(n)
        

      

    # 3순위: 클라이언트/사업명 힌트 (주소 미결정 시)
    if client_name:
        office = _assign_office_by_client_name(client_name)
        if office:
            n = dict(base_notice); n["assigned_office"] = office; n["address"] = mall_addr or ""
            #print(f"  [✅ 저장 후보] {n.get('assigned_office')} / {n.get('client')}")
            return _save(n)
        
    offices = assign_offices_by_keywords(client_name or "", base_notice.get("project_name",""))
    if offices:
        n = dict(base_notice)
        n["assigned_office"] = "/".join(offices) if isinstance(offices, (list,tuple)) else str(offices)
        n["address"] = mall_addr or ""
        #print(f"  [⚠️ 저장 후보 (제목 기반 광역)] {n.get('assigned_office')} / {n.get('client')}")
        return _save(n)

    # 최종 제외
    print_exclude_once(base_notice, client_name, mall_addr or addr or "")
    return


def finalize_notice_dict(base_notice, client_code, mall_addr, client_name):
    # 저장하지 말고 dict를 돌려줘서 벌크 업서트 경로에서 사용
    return expand_and_store_with_priority(base_notice, client_code, mall_addr, client_name, save=False)



# =========================
# 수집기
# =========================
# [수정] _build_base_notice 함수에 source_system 필드 추가
def _build_base_notice(stage: str, biz_type: str, project_name: str, client: str, phone: str,
                       model: str, qty: int, amount: str, is_cert: str, notice_date: str, detail_link: str,
                       source: str = 'G2B', kapt_code: Optional[str] = None) -> Dict: # kapt_code 파라미터 추가
    return {
        "stage": stage, "biz_type": biz_type,
        "project_name": project_name or "",
        "client": client or "",
        "address": "",
        "phone_number": phone or "",
        "model_name": model or "",
        "quantity": qty or 0,
        "amount": amount or "",
        "is_certified": is_cert or "확인필요",
        "notice_date": notice_date or "",
        "detail_link": detail_link or "",
        "assigned_office": "",
        "is_favorite": False, "status": "", "memo": "",
        "source_system": source,
        "kapt_code": kapt_code, # 반환 딕셔셔리에 kapt_code 추가
    }

# === 공공데이터포털 API 재시도 세션 ===
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def build_retry_session():
    """조달청 OpenAPI용 요청 세션(자동 재시도/백오프)"""
    retry = Retry(
        total=5,
        backoff_factor=0.6,  # 0.6s, 1.2s, 2.4s ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = build_retry_session()

def get_json_with_retry(url, params, timeout=20):
    """타임아웃 및 자동 재시도 포함 요청"""
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_kapt_basic_info(
    kapt_code: str,
    *,
    allow_non_standard: bool = False,   # KB… 같은 비표준 코드 허용 여부
    max_retries: int = 2,
    backoff_sec: float = 0.25
) -> Optional[Dict[str, Any]]:
    """단지 코드로 K-APT 기본 정보를 조회합니다. (안전가드/재시도 포함)"""
    if not kapt_code:
        return None

    # 표준 코드(A########)만 보려면 아래 가드 유지, 비표준 허용시 allow_non_standard=True로 호출
    if (not allow_non_standard) and (not re.match(r"^A\d{8}$", kapt_code)):
        return None

    params = {
        "serviceKey": _cfg("KAPT_SERVICE_KEY"),
        "kaptCode": kapt_code,
        "_type": "json",
    }

    for attempt in range(max_retries + 1):
        try:
            data = http_get_json(api_url(KAPT_BASIC_INFO_PATH), params)

            if not isinstance(data, dict):
                raise ValueError("empty/invalid response")

            resp = data.get("response") or {}
            body = resp.get("body") or {}

            item = body.get("item")
            # 일부 응답이 list로 올 수 있어 대비
            if item is None:
                items = body.get("items") or body.get("itemList")
                if isinstance(items, list) and items:
                    item = items[0]

            # 정상 dict면 반환
            if isinstance(item, dict) and item:
                return item

            # 명시적으로 '없음'인 경우 조용히 None
            total = body.get("totalCount")
            if isinstance(total, (int, str)) and int(total or 0) == 0:
                return None

            raise ValueError("no item in response")

        except Exception as e:
            if attempt < max_retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            print(f"  [Error] K-APT 기본정보 조회 실패 ({kapt_code}): {e}")
            return None
    
def _compose_display_addr(item: dict) -> str:
    """
    도로명주소가 있으면 유지하되, as3(읍/면/동) 또는 bjd_mapper 동명이 있으면 괄호 보강.
    없으면 as1+as2+as3 조합.
    """
    road = (item.get("roadAddr") or item.get("addr") or "").strip()
    as1 = (item.get("as1") or "").strip()
    as2 = (item.get("as2") or "").strip()
    as3 = (item.get("as3") or "").strip()

    # bjd 기반 동명 보강 시도
    dong_hint = ""
    try:
        from bjd_mapper import get_bjd_name
        bjd = (item.get("bjdCode") or item.get("bjd_code") or "").strip()
        if bjd:
            dong_hint = (get_bjd_name(bjd) or "").split()[-1]  # 마지막 토큰(동/읍/면)만 힌트로
    except Exception:
        pass

    # as3가 있으면 우선 사용
    if as3:
        dong_hint = as3

    if road:
        if dong_hint and dong_hint not in road:
            return f"{road} ({dong_hint})"
        return road

    combo = " ".join(x for x in (as1, as2, as3) if x)
    return combo

def _extract_kapt_phone(basic: dict | None) -> str:
    """
    K-APT 기본정보에서 관리사무소 전화 추출(키 변동 안전).
    숫자/하이픈만 남기고 0~9/-(최대 하나씩) 정규화.
    """
    if not isinstance(basic, dict):
        return ""
    # 필드 후보(변동 대응)
    CAND_KEYS = [
        "mngTel", "mngTelNo", "mngTelno", "kaptTel", "telNo",
        "officeTel", "officeTelNo", "managerTel",
        "asTel", "as1Tel", "as2Tel", "as3Tel",
        "tel", "phone"
    ]
    raw = ""
    for k in CAND_KEYS:
        v = basic.get(k)
        if v and isinstance(v, str) and v.strip():
            raw = v.strip()
            break
    if not raw:
        return ""

    import re
    # 숫자/하이픈 외 제거
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        return ""
    # 8~11자리 케이스 정규화(대구 지역 국번/휴대 포함)
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    if len(digits) == 9:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 10:
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}" if digits.startswith(("02",)) else f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    # 그 외는 최대한 하이픈 삽입 못하면 원본 반환
    return raw



def fetch_kapt_maintenance_history(kapt_code: str) -> list[dict]:
    """
    K-APT 유지관리 이력 조회 (안전 정규화)
    - 'list has no attribute get' 예외를 원천 차단
    """
    url = api_url(KAPT_MAINTENANCE_PATH)
    params = {
        "serviceKey": _cfg("KAPT_SERVICE_KEY"),
        "pageNo": "1",
        "numOfRows": "100",
        "kaptCode": (kapt_code or "").strip(),
        "type": "json"
    }
    data = http_get_json(url, params)
    rows = _kapt_items_safely(data)  # ← 안전 정규화

    out = []
    for r in rows:
        out.append({
            "parentParentName": _as_text(r.get("parentParentName")),
            "parentName": _as_text(r.get("parentName")),
            "mnthEtime": _as_text(r.get("mnthEtime")),
            "year": _as_text(r.get("year")),
            "useYear": _as_text(r.get("useYear")),
        })
    return out



def fetch_and_process_kapt_bids(search_ymd: str):
    """K-APT 입찰공고 수집 — 로그는 '제외/저장 대기/일괄 저장'만 출력 (주소는 로그 끝에 표시)."""
    print(f"\n--- [{to_ymd(search_ymd)}] 공동주택(K-APT) 입찰공고 수집 ---")

    try:
        params_first = {
            "serviceKey": _cfg("KAPT_SERVICE_KEY"), "pageNo": "1", "numOfRows": "1",
            "startDate": search_ymd, "endDate": search_ymd, "_type": "json"
        }
        first = http_get_json(api_url(KAPT_BID_LIST_PATH), params_first)
        total = int(first.get("response", {}).get("body", {}).get("totalCount", 0))
        if total == 0:
            print("- 데이터 없음"); return
    except Exception as e:
        print(f"[Error] K-apt 총 건수 조회 실패: {e}"); return

    page_size = 100
    total_pages = (total + page_size - 1) // page_size
    #print(f"- 총 {total}건 / {total_pages}p")
    print(f"- 총 {total}건")

    params_list = [{
        "serviceKey": _cfg("KAPT_SERVICE_KEY"), "pageNo": str(p), "numOfRows": str(page_size),
        "startDate": search_ymd, "endDate": search_ymd, "_type": "json"
    } for p in range(1, total_pages + 1)]

    pages = fetch_pages_parallel(api_url(KAPT_BID_LIST_PATH), params_list)

    buffer = []
    for data in pages:
        items = data.get("response", {}).get("body", {}).get("items", [])
        if not items:
            continue

        for it in items:
            title = (it.get("bidTitle") or "").strip()
            if not is_relevant_text(title,
                                    _as_text(it.get("codeClassifyType1")),
                                    _as_text(it.get("codeClassifyType2")),
                                    _as_text(it.get("codeClassifyType3")),
                                    _as_text(it.get("bidMethod")),
                                    _as_text(it.get("bidKaptname"))):
                continue


            kapt_code   = (it.get("aptCode") or "").strip()
            client_name = (it.get("bidKaptname") or "").strip() or "단지명 없음"
            bid_no      = it.get("bidNum")
            detail_link = f"https://www.k-apt.go.kr/bid/bidDetail.do?bid_noti_no={bid_no}" if bid_no else ""
            biz_type    = it.get("codeClassifyType1", "기타")

            # 주소/법정동코드 보강
            bjd_code, addr_txt = "", ""
            if kapt_code:
                basic = fetch_kapt_basic_info(kapt_code)
                if basic:
                    bjd_code = str(basic.get("bjdCode") or "")
                    addr_txt = (basic.get("doroJuso") or basic.get("kaptAddr") or "").strip()
            if not bjd_code:
                raw = str(it.get("bidArea") or "")
                if len(raw) >= 8:
                    bjd_code = raw[:10]
            if not addr_txt and bjd_code and HAS_BJD_MAPPER:
                addr_txt = get_bjd_name(bjd_code)

            # 관할지사 판정
            assigned_office = _assign_office_from_bjd_code(bjd_code, addr_txt)

            # 대구권 외면 제외
            if assigned_office.startswith("관할") and not (addr_txt.startswith("대구") or bjd_code.startswith("27")):
                log_kapt_excluded(client_name, addr_txt or bjd_code)
                continue

            base = _build_base_notice(
                stage="입찰공고",
                biz_type=biz_type,
                project_name=title,
                client=client_name,
                phone="", model="", qty=0, amount="",
                is_cert="확인필요",
                notice_date=to_ymd(it.get("bidRegDate")),
                detail_link=detail_link,
                source='K-APT',
                kapt_code=kapt_code,
            )
            base["assigned_office"] = assigned_office

            n = finalize_notice_dict(base, None, addr_txt, client_name)
            if n:
                buffer.append(n)
                log_kapt_pending(assigned_office, client_name, addr_txt or bjd_code)

    if buffer:
        bulk_upsert_notices(buffer)
        log_kapt_bulk_saved(len(buffer))


# [추가] K-APT 입찰 결과 API 엔드포인트
KAPT_BID_RESULT_PATH = "/1613000/ApHusBidResultNoticeInfoOfferServiceV2/getPblAncDeSearchV2"





def _assign_office_from_bjd_code(bjd_code: str, addr_text: str = "") -> str:
    """
    나라장터 기준 관할 추정 (문자열 반환)
    - 포항(단순화): 북구+동/읍/면이면 흥해/송라/신광/청하/기계/기북/죽장 → 북포항, 그 외 포항
      북구(구까지만) → '포항지사/북포항지사', 북구 외 포항 → 포항지사
    - 대구: 중구/북구=직할, 동구/수성구=동대구, 서구/남구=서대구
      달서구: 동레벨 기본 남대구(감삼/두류/본리/성당/죽전만 서대구), 구까지만 '남대구지사/서대구지사'
      달성군: 다사/하빈=서대구, 가창=동대구, 그 외 동/읍/면=남대구, 군까지만 '남대구지사/서대구지사'
    - 경북 기타: 경주/경산/김천/영천/칠곡/성주/청도/고령/영덕
    - 실패 → '관할지사확인요망'
    """
    def _has_dong_level(a: str) -> bool:
        return bool(re.search(r"(동|읍|면|리)\b", a or ""))

    # 주소를 반드시 문자열로 확보 (순환의존 없이)
    addr = resolve_address_from_bjd(addr_text=addr_text, bjd_code=bjd_code)
    if not addr:
        return "관할지사확인요망"
    
    # ─ 대구
    if ("대구" in addr) or ("대구광역시" in addr):
        has_dong = _has_dong_level(addr)
        if ("중구" in addr) or ("북구" in addr):
            return "직할"
        if ("동구" in addr) or ("수성구" in addr):
            return "동대구지사"
        if ("서구" in addr) or ("남구" in addr):
            return "서대구지사"
        if "달서구" in addr:
            if has_dong:
                if any(d in addr for d in ["감삼","두류","본리","성당","죽전"]):
                    return "서대구지사"
                return "남대구지사"
            return "남대구지사/서대구지사"
        if "달성군" in addr:
            if has_dong:
                if any(e in addr for e in ["다사읍","하빈면"]):
                    return "서대구지사"
                if "가창면" in addr:
                    return "동대구지사"
                return "남대구지사"
            return "남대구지사/서대구지사"
        return "직할"



    # ─ 포항
    if (("포항" in addr) or ("포항시" in addr)) and not ("대구광역시" in addr or "대구시" in addr):
        if re.search(r"포항시\s*북구", addr):
            if _has_dong_level(addr):
                if any(s in addr for s in ["흥해", "송라", "신광", "청하", "기계", "기북", "죽장"]):
                    return "북포항지사"
                return "포항지사"
            return "포항지사/북포항지사"
        return "포항지사"




    # ─ 경북 기타
    mapping = {
        "경주": "경주지사","경산": "경산지사","김천": "김천지사","영천": "영천지사",
        "칠곡": "칠곡지사","성주": "성주지사","청도": "청도지사","고령": "고령지사","영덕": "영덕지사",
    }
    for key, office in mapping.items():
        if key in addr:
            return office

    return "관할지사확인요망"

# ===== 공용 상수/도우미 =====
PAGE_SIZE = 100  # 전역 통일

# 엔드포인트
PATH_PBL  = "/1613000/ApHusBidResultNoticeInfoOfferServiceV2/getPblAncDeSearchV2"   # 공고일 범위
PATH_CLOS = "/1613000/ApHusBidResultNoticeInfoOfferServiceV2/getBidClosDeSearchV2"  # 마감일 범위
PATH_STTS = "/1613000/ApHusBidResultNoticeInfoOfferServiceV2/getBidSttusSearchV2"   # 상태+연도

from collections import Counter
import re, csv, os
from datetime import datetime, timedelta

def _as_ymd8(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _parse_ymd8(s: str) -> datetime:
    s = "".join(ch for ch in str(s) if ch.isdigit())[:8]
    return datetime.strptime(s, "%Y%m%d")
from datetime import datetime, timedelta

def _is_business_day(d: datetime) -> bool:
    return d.weekday() < 5  # 월(0)~금(4)

def prev_business_day(yyyymmdd: str) -> str:
    d = datetime.strptime(yyyymmdd, "%Y%m%d")
    while True:
        d -= timedelta(days=1)
        if _is_business_day(d):
            return d.strftime("%Y%m%d")

def next_business_day(yyyymmdd: str) -> str:
    d = datetime.strptime(yyyymmdd, "%Y%m%d")
    while True:
        d += timedelta(days=1)
        if _is_business_day(d):
            return d.strftime("%Y%m%d")

def _date8(s: str) -> str:
    if not s:
        return ""
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return d[:8] if len(d) >= 8 else ""

# ★ apt_list.csv 매핑 로더 (유연한 헤더 대응)
#   - config.KAPT_APT_LIST_PATH 가 있으면 우선 사용, 없으면 실행 경로 기준 탐색
_APT_MAP = None  # 캐시

def _load_apt_map():
    global _APT_MAP
    if _APT_MAP is not None:
        return _APT_MAP

    path = _cfg("KAPT_APT_LIST_PATH")

    if not path:
        # 흔한 위치 후보들
        for cand in [
            "./apt_list.csv",
            "./data/apt_list.csv",
            "C:/bh/_ing/kapt/apt_list.csv",
            "C:/bh/_final/kapt/apt_list.csv",
        ]:
            if os.path.exists(cand):
                path = cand
                break

    amap = {}
    if not path or not os.path.exists(path):
        print(f"[K-apt][관할] apt_list.csv 미발견 → bidArea 폴백 사용")
        _APT_MAP = amap
        return amap

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            # 유연 헤더 추출
            # 예시 헤더 후보: aptCode / 단지코드, addr / address / 주소, bjd_code / 법정동코드
            for row in rdr:
                code = row.get("aptCode") or row.get("단지코드") or row.get("kaptCode") or ""
                code = str(code).strip()
                if not code:
                    continue
                addr = row.get("addr") or row.get("address") or row.get("주소") or ""
                bjd  = row.get("bjd_code") or row.get("법정동코드") or row.get("bjdCode") or ""
                amap[code] = {
                    "addr": str(addr).strip(),
                    "bjd_code": str(bjd).strip(),
                }
        print(f"[K-apt][관할] apt_list.csv 로드: {len(amap)}건")
    except Exception as e:
        print(f"[K-apt][관할] apt_list.csv 로드 실패: {type(e).__name__}: {e}")
        amap = {}

    _APT_MAP = amap
    return amap

def _resolve_office_by_apt_or_bidarea(kapt_code: str, bid_area: str):
    """
    1순위: 단지코드 매핑(정확 주소/법정동)
    2순위: bidArea(시·도 코드, 예: 대구=27)
    """
    amap = _load_apt_map()
    addr_txt, bjd_code = "", ""
    if kapt_code and kapt_code in amap:
        addr_txt = amap[kapt_code].get("addr", "") or ""
        bjd_code = amap[kapt_code].get("bjd_code", "") or ""
    if not bjd_code:
        bjd_code = str(bid_area or "")

    # addr_txt 없고 대구 시·도코드라면 최소 표시
    if not addr_txt and bjd_code.startswith("27"):
        addr_txt = "대구"

    assigned = _assign_office_from_bjd_code(bjd_code, addr_txt)
    return assigned, addr_txt, bjd_code



def fetch_and_process_kapt_bid_results(search_ymd: str):
    """
    K-APT 입찰결과 수집
      - 1순위: 당일(마감/공고)
      - 실패 시: 상태+연도 전수 → 당일 ±1영업일 보정 필터
      - 키워드(가변 인자) + 관할 + 중복 제거
      - 사용자 표시 로그: 나라장터 톤(총 N건 / Pp, 일괄 저장 N건 / 데이터 없음)
    """
    print(f"\n--- [{to_ymd(search_ymd)}] 공동주택(K-APT) 입찰결과 수집 ---")
    svc_key = getattr(_local_config, "KAPT_SERVICE_KEY_DECODING", None) or _cfg("KAPT_SERVICE_KEY")

    def _first_page(url_path, tag):
        try:
            r = http_get_json(api_url(url_path), {
                "serviceKey": svc_key, "_type": "json",
                "pageNo": "1", "numOfRows": "1",
                "startDate": search_ymd, "endDate": search_ymd,
            })
            resp = (r or {}).get("response") or {}
            header = resp.get("header") or {}
            body = resp.get("body") or {}
            code = header.get("resultCode")
            msg  = header.get("resultMsg")
            total = int(body.get("totalCount", 0) or 0)
            _debug(f"· [{tag}] resultCode={code}, totalCount={total}, msg={msg}")
            return total
        except Exception as e:
            _debug(f"· [{tag}] 요청 실패: {type(e).__name__}: {e}")
            return 0

    # 1) 당일 수집(마감일/공고일)
    total_clos = _first_page(PATH_CLOS, "마감일")
    total_pbl  = _first_page(PATH_PBL,  "공고일")

    pages_all = []

    def _fetch_pages(url_path, total, tag):
        if total <= 0:
            return []
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        #_debug(f"· [{tag}] 총 {total}건 / {total_pages}p → 병렬 수집")
        _debug(f"· [{tag}] 총 {total}건")
        plist = [{
            "serviceKey": svc_key, "_type": "json",
            "pageNo": str(p), "numOfRows": str(PAGE_SIZE),
            "startDate": search_ymd, "endDate": search_ymd,
        } for p in range(1, total_pages + 1)]
        try:
            return fetch_pages_parallel(api_url(url_path), plist)
        except Exception as e:
            _debug(f"· [{tag}] 페이지 수집 실패: {type(e).__name__}: {e}")
            return []

    if total_clos > 0:
        pages_all.extend(_fetch_pages(PATH_CLOS, total_clos, "마감일"))
    if total_pbl  > 0:
        pages_all.extend(_fetch_pages(PATH_PBL,  total_pbl,  "공고일"))

    # 사용자 요약(당일 합산)
    sum_total = (total_clos or 0) + (total_pbl or 0)
    if sum_total > 0:
        _print_total_summary(sum_total)  # "- 총 N건 / Pp"

    # 2) 상태+연도 전수 → 보정필터
    if not pages_all:
        _debug("· [상태+연도] 보정 수집 시도 (4=유찰, 5=낙찰)")

        def _collect_state(state):
            r = http_get_json(api_url(PATH_STTS), {
                "serviceKey": svc_key, "_type": "json",
                "pageNo": "1", "numOfRows": "1",
                "bidState": state, "searchYear": search_ymd[:4],
            })
            resp = (r or {}).get("response") or {}
            header = resp.get("header") or {}
            body = resp.get("body") or {}
            code = header.get("resultCode")
            total = int(body.get("totalCount", 0) or 0)
            _debug(f"· [상태+연도(state={state})] resultCode={code}, totalCount={total}, msg={header.get('resultMsg')}")
            if total <= 0:
                return []
            total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            #_debug(f"· [상태+연도(state={state})] 총 {total}건 / {total_pages}p → 병렬 수집")
            _debug(f"· [상태+연도(state={state})] 총 {total}건")
            plist = [{
                "serviceKey": svc_key, "_type": "json",
                "pageNo": str(p), "numOfRows": str(PAGE_SIZE),
                "bidState": state, "searchYear": search_ymd[:4],
            } for p in range(1, total_pages + 1)]
            return fetch_pages_parallel(api_url(PATH_STTS), plist)

        st_pages = []
        st_pages.extend(_collect_state("5"))
        st_pages.extend(_collect_state("4"))

        raw_items = []
        for pg in st_pages:
            raw_items.extend((((pg or {}).get("response") or {}).get("body") or {}).get("items") or [])

        # 분포/진단은 VERBOSE에서만
        clos_ctr = Counter(_date8(it.get("bidDeadline")) for it in raw_items if _date8(it.get("bidDeadline")))
        pbl_ctr  = Counter(_date8(it.get("bidRegdate"))  for it in raw_items if _date8(it.get("bidRegdate")))
        _debug(f"· [상태+연도 전체] 총 {len(raw_items)}건 | 빈 마감일:{len(raw_items)-sum(clos_ctr.values())}, 빈 공고일:{len(raw_items)-sum(pbl_ctr.values())}")

        alt_days = {search_ymd, prev_business_day(search_ymd), next_business_day(search_ymd)}
        filtered = [it for it in raw_items
                    if _date8(it.get("bidDeadline")) in alt_days
                    or _date8(it.get("bidRegdate"))  in alt_days]

        _print_total_summary(len(filtered), tag="")  # 사용자 요약: "- 총 N건 / Pp (보정)"

        if not filtered:
            _print_data_none()
            return

        pages_all = [{"response": {"body": {"items": filtered}}}]

    # 3) 파싱 +(키워드/관할/중복) 정제
    seen_bidnum = set()
    buffer = []
    stats = dict(total_items=0, after_kw=0, after_region=0, dedup_skip=0, excluded_region=0)
    EXCLUDE_LOG_MAX = 50
    excl_shown = 0

    for data in pages_all:
        items = (((data or {}).get("response") or {}).get("body") or {}).get("items") or []
        stats["total_items"] += len(items)
        for it in items:
            title = (it.get("bidTitle") or "").strip()
            state = (it.get("bidState") or "").strip()
            stage = "계약완료" if state == "5" else ("입찰결과(유찰)" if state == "4" else "기타")
            biz_type = it.get("codeClassifyType1", "기타")
            kapt_code = (it.get("aptCode") or "").strip()
            client_name = (it.get("bidKaptname") or "").strip() or "단지명 없음"
            bid_no = (it.get("bidNum") or "").strip()

            # 키워드 필터(다필드)
            biz1 = _as_text(it.get("codeClassifyType1"))
            biz2 = _as_text(it.get("codeClassifyType2"))
            biz3 = _as_text(it.get("codeClassifyType3"))
            bid_method = _as_text(it.get("bidMethod"))
            kapt_name  = _as_text(it.get("bidKaptname"))
            if not _pass_keyword_filter(title, biz1, biz2, biz3, bid_method, kapt_name):
                continue
            stats["after_kw"] += 1

            # --- [FIX] 관할/주소/법정동코드 초기 세팅 ---
            # 1) apt_list.csv 우선 조회
            addr_txt, pre_office, bjd_code = lookup_apt_by_code(kapt_code)

            # 2) 모자라면 K-APT 기본정보로 보강
            if not (addr_txt and bjd_code):
                basic = fetch_kapt_basic_info(kapt_code) or {}
                # 기본정보의 주소/법정동코드 필드명은 시스템 상황에 따라 다를 수 있어 유연하게 처리
                addr_txt = addr_txt or (basic.get("address") or basic.get("addr") or "")
                bjd_code = bjd_code or (basic.get("bjdCode") or basic.get("bjd_code") or "")

            # 3) 그래도 법정동코드가 없으면 bidArea(지역코드)라도 폴백
            bjd_code = (bjd_code or str(it.get("bidArea") or "")).strip()

            # 4) 초기 관할 추정 (apt_list의 pre_office가 있으면 우선, 없으면 법정동/주소로 계산)
            assigned = (pre_office or _assign_office_from_bjd_code(bjd_code, addr_txt) or "").strip()

            # --- 기존 코드 ---
            # 관할 판정 (aptMap 우선 → bidArea 폴백)
            assigned, addr_txt, bjd_code = _narrow_office_with_basic_info(assigned, kapt_code, addr_txt, bjd_code)


         
            if assigned.startswith("관할") and not (bjd_code.startswith("27") or (addr_txt and addr_txt.startswith("대구"))):
                stats["excluded_region"] += 1
                if excl_shown < EXCLUDE_LOG_MAX:
                    log_kapt_excluded(client_name, "-")
                    excl_shown += 1
                continue
            stats["after_region"] += 1

            # 중복 제거
            if bid_no and bid_no in seen_bidnum:
                stats["dedup_skip"] += 1
                continue
            if bid_no:
                seen_bidnum.add(bid_no)

            detail = f"https://www.k-apt.go.kr/bid/bidResultDetail.do?bid_noti_no={bid_no}" if bid_no else ""

            base = _build_base_notice(
                stage=stage, biz_type=biz_type, project_name=title,
                client=client_name, phone="", model="", qty=0,
                amount=(it.get("amount") or ""), is_cert="확인필요",
                notice_date=to_ymd(it.get("bidDeadline") or it.get("bidRegdate")),
                detail_link=detail, source='K-APT', kapt_code=kapt_code
            )
            base["assigned_office"] = assigned
            base["address"] = addr_txt or base.get("address","") 
            n = finalize_notice_dict(base, None, addr_txt, client_name)
            if n:
                buffer.append(n)

    # 4) 저장/요약 로그
    if not buffer:
        _debug(f"(수집:{stats['total_items']}, 키워드후:{stats['after_kw']}, "
               f"관할후:{stats['after_region']}, 중복제외:{stats['dedup_skip']}, "
               f"타지역제외:{stats['excluded_region']})")
        _print_data_none()
        return

    try:
        bulk_upsert_notices(buffer)
        _print_bulk_saved(len(buffer))  # "  [✅ 일괄 저장] N건"
    except Exception as e:
        print(f"  [Error] 저장 실패: {type(e).__name__}: {e} (후보:{len(buffer)})")


def _collect_by_state_year(bid_state: str, year: str):
    svc_key = (
        getattr(_local_config, "KAPT_SERVICE_KEY_DECODING", None)
        if _local_config else None
    ) or _cfg("KAPT_SERVICE_KEY")

    first = http_get_json(api_url(PATH_STTS), {   # <-- 여기
        "serviceKey": svc_key,
        "pageNo": "1", "numOfRows": "1",
        "bidState": bid_state, "searchYear": year,
        "_type": "json",
    })
    total = int(((first.get("response") or {}).get("body") or {}).get("totalCount", 0) or 0)
    if total == 0:
        return []

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    params_list = [{
        "serviceKey": svc_key,
        "pageNo": str(p), "numOfRows": str(PAGE_SIZE),
        "bidState": bid_state, "searchYear": year, "_type": "json"
    } for p in range(1, total_pages + 1)]

    pages = fetch_pages_parallel(api_url(PATH_STTS), params_list)  # <-- 여기
    results = []
    for data in (pages or []):
        items = (((data or {}).get("response") or {}).get("body") or {}).get("items")
        if not items:
            continue
        if isinstance(items, list):
            results.extend(items)
        else:
            results.append(items)  # 단일 dict 방어
    return results

from datetime import datetime, timedelta
from calendar import monthrange

def _month_chunks(start_ymd: str, end_ymd: str):
    s = datetime.strptime(start_ymd, "%Y%m%d")
    e = datetime.strptime(end_ymd, "%Y%m%d")
    cur = datetime(s.year, s.month, 1)
    while cur <= e:
        last_day = monthrange(cur.year, cur.month)[1]
        chunk_start = max(s, cur)
        chunk_end   = min(e, datetime(cur.year, cur.month, last_day))
        yield chunk_start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        # 다음 달 1일
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)

def _count_private_contracts(svc_key, start_ymd, end_ymd):
    q = {"serviceKey": svc_key, "_type": "json", "pageNo": "1", "numOfRows": "1",
         "startDate": start_ymd, "endDate": end_ymd}
    r = http_get_json(api_url(KAPT_PRIVATE_CONTRACT_PATH), q)
    body = ((r or {}).get("response") or {}).get("body") or {}
    return int(body.get("totalCount", 0) or 0)

from datetime import datetime

def fetch_and_process_kapt_private_contracts(search_ymd: str):
    """K-APT 수의계약 공지 수집 (시스템 표준: regDate 기준 정렬/신규/저장)."""
    # === 날짜 범위 계산 ===
    endDate = search_ymd  # 조회 종료일 = 입력일 (YYYYMMDD)
    end_dt = datetime.strptime(search_ymd, "%Y%m%d")
    startDate = f"{end_dt.year - 1}0101"  # 직전년도 1/1 ~ 조회일까지

    print(f"\n--- [K-APT 수의계약] 조회기간: {startDate} ~ {endDate} ---")

    svc_key = getattr(_local_config, "KAPT_SERVICE_KEY_DECODING", None) or _cfg("KAPT_SERVICE_KEY")
    PAGE = PAGE_SIZE

    # yyyymmdd -> yyyy-mm-dd
    def _dash(yyyymmdd: str) -> str:
        s = (yyyymmdd or "").strip()
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s

    def _try_count(params, tag: str) -> int:
        r = http_get_json(api_url(KAPT_PRIVATE_CONTRACT_PATH), params)
        body = ((r or {}).get("response") or {}).get("body") or {}
        total = int(body.get("totalCount", 0) or 0)
        _debug(
            f"· [{tag}] resultCode={((r or {}).get('response') or {}).get('header',{}).get('resultCode')} "
            f"totalCount={total}, msg={((r or {}).get('response') or {}).get('header',{}).get('resultMsg')}"
        )
        return total

    # === regDate 기준 전체 건수 조회 ===
    q1 = {
        "serviceKey": svc_key,
        "_type": "json",
        "pageNo": "1",
        "numOfRows": "1",
        "startDate": startDate,   # regDate 시작(YYYYMMDD)
        "endDate": endDate        # regDate 종료(YYYYMMDD)
    }
    total = _try_count(q1, f"등록일 {startDate}~{endDate}")

    if total == 0:
        print("  - 데이터 없음")
        return

    # === 페이징 ===
    total_pages = (total + PAGE - 1) // PAGE
    _debug(f"· [수의계약] 총 {total}건 / {total_pages}p → 병렬 수집")
    params_list = [{**q1, "pageNo": str(p), "numOfRows": str(PAGE)} for p in range(1, total_pages + 1)]

    # === 병렬 수집 ===
    pages = fetch_pages_parallel(api_url(KAPT_PRIVATE_CONTRACT_PATH), params_list)

    # === 파싱 및 저장 ===
    buffer, stats = [], dict(total_items=0, after_kw=0, after_region=0)

    # 상세보기 링크: bidDetail.do (시스템과 동일 UX)
    def _make_detail_link(pc_num: str, kapt_code: str | None) -> str:
        if not pc_num:
            return ""
        return (
            "https://www.k-apt.go.kr/bid/bidDetail.do"
            f"?searchBidGb=private_contract"
            f"&bidTitle=&aptName=&searchDateGb=reg"
            f"&dateStart={_dash(startDate)}&dateEnd={_dash(endDate)}"
            f"&dateArea=1&bidState=&codeAuth=&codeWay=&codeAuthSub=&codeSucWay="
            f"&codeClassifyType1=&codeClassifyType2=&codeClassifyType3="
            f"&pageNo=1&type=4&bidArea=&bidNum={pc_num}"
            f"&bidNo=&mainKaptCode=&aptCode={(kapt_code or '')}"
        )

    for data in pages:
        items = (((data or {}).get("response") or {}).get("body") or {}).get("items") or []
        if isinstance(items, dict):
            items = [items]
        stats["total_items"] += len(items)

        for it in items:
            title         = (it.get("pcTitle") or "").strip()
            pc_date_raw   = (it.get("pcDate") or "").strip()    # 계약일자 (참고)
            pc_st_raw     = (it.get("pcStDate") or "").strip()  # 시작
            pc_ed_raw     = (it.get("pcEdDate") or "").strip()  # 종료
            reg_date_raw  = (it.get("regDate") or "").strip()   # 등록일 (시스템 기준)
            kapt_code     = it.get("kaptCode")
            client_name   = it.get("kaptName") or "단지명 없음"
            mall_addr     = (it.get("area") or "").strip()
            contract_no   = (it.get("pcNum") or "").strip()

            # 텍스트 필터
            if not is_relevant_text(
                title,
                _as_text(it.get("pcReason", "")),
                _as_text(it.get("codeClassifyType1", "")),
            ):
                continue
            stats["after_kw"] += 1

            detail_link = _make_detail_link(contract_no, kapt_code)

            # ★ 시스템 얼라인: 저장 기준일 = regDate
            notice_date = to_ymd(reg_date_raw)

            # 필요 시 계약정보를 memo에 보존 (DB 스키마가 contract_date 없음 가정)
            extra_memo = ""
            if pc_date_raw or pc_st_raw or pc_ed_raw:
                extra_memo = f"[계약일]{pc_date_raw} [기간]{pc_st_raw}~{pc_ed_raw}".strip()

            base = _build_base_notice(
                stage="수의계약",
                biz_type=(it.get("codeClassifyType1", "") or "기타"),
                project_name=title,
                client=client_name,
                phone=(it.get("companyTel", "") or ""),
                model="",
                qty=0,
                amount=_as_text(it.get("pcAmount", "")),
                is_cert="확인필요",
                notice_date=notice_date,         # ← regDate 고정
                detail_link=detail_link,
                source="K-APT",
                kapt_code=kapt_code
            )

            # finalize에서 memo를 합치고 싶다면 여기서 merge (옵션)
            if extra_memo:
                base["memo"] = (base.get("memo") or "")
                if base["memo"]:
                    base["memo"] += "\n"
                base["memo"] += extra_memo

            n = finalize_notice_dict(base, None, mall_addr, client_name)
            if n:
                stats["after_region"] += 1
                buffer.append(n)

    if not buffer:
        _debug(f"(수집:{stats['total_items']}, 키워드후:{stats['after_kw']}, 관할후:{stats['after_region']})")
        print("  - 데이터 없음")
        return

    try:
        bulk_upsert_notices(buffer)
        print(f"  [✅ 일괄 저장] {len(buffer)}건")
        _debug(f"(수집:{stats['total_items']}, 키워드후:{stats['after_kw']}, 관할후:{stats['after_region']})")
    except Exception as e:
        print(f"  [Error] 저장 실패: {type(e).__name__}: {e} (후보:{len(buffer)})")

# --- 발주계획 ---
def fetch_and_process_order_plans(search_ymd: str):
    print(f"\n--- [{to_ymd(search_ymd)}] 발주계획(나라장터) 수집 ---")

    # 1) 총건수 1회 조회
    first = http_get_json(api_url(ORDER_PLAN_LIST_PATH), {
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": "1", "numOfRows": "1",
        "inqryDiv": "1", "inqryBgnDt": f"{search_ymd}0000", "inqryEndDt": f"{search_ymd}2359"
    })
    body = _as_dict(first.get("response", {}).get("body"))
    total = int(body.get("totalCount", 0))
    if total == 0:
        print("  - 데이터 없음"); return

    page_size   = 100
    total_pages = (total + page_size - 1) // page_size
    #print(f"  - 총 {total}건 / {total_pages}p")
    print(f"  - 총 {total}건")

    # 2) 페이지 병렬 요청
    params_list = [{
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": str(p), "numOfRows": str(page_size),
        "inqryDiv": "1", "inqryBgnDt": f"{search_ymd}0000", "inqryEndDt": f"{search_ymd}2359"
    } for p in range(1, total_pages + 1)]

    pages = fetch_pages_parallel(api_url(ORDER_PLAN_LIST_PATH), params_list)

    # 3) 페이지 결과 처리 (메인 스레드에서만 DB 저장)
    for data in pages:
        items = _as_items_list(_as_dict(data.get("response", {}).get("body")))
        for it in items:
            if it.get("bsnsDivNm") != "물품":
                continue
            title = it.get("bizNm", "")
            if not is_relevant_text(title,
                                    _as_text(it.get("bsnsDivNm")),
                                    _as_text(it.get("itemNm") or it.get("prdctNm")),
                                    _as_text(it.get("dminsttNm") or it.get("dmndInsttNm"))):
                continue


            client_code = it.get("orderInsttCd") or it.get("dminsttCd")
            client_name = it.get("orderInsttNm") or it.get("dminsttNm") or "기관명 없음"
            mall_addr   = guess_mall_addr(it)

            plan_no = it.get('orderPlanUntyNo') or ''
            detail_link = f"https://www.g2b.go.kr/pt/menu/selectSubFrame.do?framesrc=/pt/orderplan/orderPlanDetail.do?orderPlanNo={plan_no}" if plan_no else ""

            base = _build_base_notice(
                "발주계획", "물품", title, client_name, it.get("telNo", ""),
                "계획 단계 확인", 0, it.get("sumOrderAmt") or "", "확인필요",
                to_ymd(it.get("nticeDt")), detail_link
            )
            expand_and_store_with_priority(base, client_code, mall_addr, client_name)


# --- 입찰공고 ---
def fetch_and_process_bid_notices(search_ymd: str):
    print(f"\n--- [{to_ymd(search_ymd)}] 입찰공고(나라장터)) 수집 ---")
    page, page_size, total_pages = 1, 100, 1
    while page <= total_pages:
        params = {
            "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
            "pageNo": str(page), "numOfRows": str(page_size),
            "bidNtceBgnDt": f"{search_ymd}0000", "bidNtceEndDt": f"{search_ymd}2359"
        }
        try:
            data = http_get_json(api_url(BID_LIST_PATH), params)
            if not isinstance(data, dict):
                print("  - 응답 없음(네트워크/서버 응답 오류). 재시도 또는 다음 페이지로 진행")
                page += 1
                time.sleep(0.35)
                continue            
            
            
            body = _as_dict(data.get("response", {}).get("body"))
            if page == 1:
                total = int(body.get("totalCount", 0))
                if total == 0:
                    print("  - 데이터 없음")
                    break
                total_pages = (total + page_size - 1) // page_size
                #print(f"  - 총 {total}건 / {total_pages}p")
                print(f"  - 총 {total}건")

            items = _as_items_list(body)
            if not items:
                page += 1
                time.sleep(0.35)
                continue

            for it in items:
                if it.get("bsnsDivNm") and it.get("bsnsDivNm") != "물품":
                    continue
                title = it.get("bidNtceNm", "") or it.get("bidNm", "")
                if not is_relevant_text(title,
                                        _as_text(it.get("bsnsDivNm")),
                                        _as_text(it.get("itemNm") or it.get("prdctNm")),
                                        _as_text(it.get("dminsttNm") or it.get("dmndInsttNm"))):
                    continue


                client_code = it.get("dmndInsttCd") or it.get("dminsttCd")
                client_name = it.get("dmndInsttNm") or it.get("dminsttNm") or "기관명 없음"
                mall_addr = guess_mall_addr(it)

                # 상세 URL
                detail_link = it.get("bidNtceUrl")
                if not detail_link:
                    bid_no = it.get('bidNtceNo') or ''
                    if bid_no:
                        detail_link = f"http://www.g2b.go.kr/pt/menu/selectSubFrame.do?framesrc=/pt/bid/bidInfoList.do?taskClCd=1&bidno={bid_no}"

                base = _build_base_notice(
                    "입찰공고", "물품", title, client_name, it.get("dmndInsttOfclTel", "") or it.get("telNo",""),
                    "공고 확인 필요", 0, it.get("asignBdgtAmt") or "", "확인필요",
                    to_ymd(it.get("bidNtceDate") or it.get("ntceDt")), detail_link or ""
                )
                expand_and_store_with_priority(base, client_code, mall_addr, client_name)

            page += 1
            time.sleep(0.35)
        except Exception as e:
            session.rollback()
            print(f"  [Error] 입찰공고 처리 오류: {e}")
            break


# --- 계약완료 ---
def fetch_and_process_contracts(search_ymd: str):
    from datetime import datetime, timedelta
    print(f"\n--- [{to_ymd(search_ymd)}] 계약완료(나라장터) 수집 ---")

    start_dt = f"{search_ymd}0000"
    end_dt = f"{search_ymd}2359"

    # 1) 총건수 1회 조회
    first = http_get_json(api_url(CNTRCT_LIST_PATH), {
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": "1", "numOfRows": "1",
        "inqryDiv": "1", "inqryBgnDt": start_dt, "inqryEndDt": end_dt
    })
    body = _as_dict(first.get("response", {}).get("body"))
    total = int(body.get("totalCount", 0))
    if total == 0:
        print("  - 데이터 없음"); return

    page_size   = 100
    total_pages = (total + page_size - 1) // page_size
    #print(f"  - 총 {total}건 / {total_pages}p")
    print(f"  - 총 {total}건")

    # 2) 파라미터 묶음 생성 후 병렬 조회
    params_list = [{
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": str(p), "numOfRows": str(page_size),
        "inqryDiv": "1", "inqryBgnDt": start_dt, "inqryEndDt": end_dt
    } for p in range(1, total_pages + 1)]

    pages = fetch_pages_parallel(api_url(CNTRCT_LIST_PATH), params_list)

    # 3) 페이지 결과 처리 (벌크업서트용 버퍼)
    buffer = []
    for data in pages:
        items = _as_items_list(_as_dict(data.get("response", {}).get("body")))
        for it in items:
            title = it.get("cntrctNm", "") or it.get("contNm","")
            if not is_relevant_text(title,
                                    _as_text(it.get("bsnsDivNm")),
                                    _as_text(it.get("itemNm") or it.get("prdctNm")),
                                    _as_text(it.get("dminsttNm") or it.get("dmndInsttNm"))):
                continue

            dm_cd = it.get("dminsttCd") or it.get("dmndInsttCd")
            cn_cd = it.get("cntrctInsttCd") or it.get("insttCd")
            client_code = dm_cd or cn_cd
            client_name = it.get("dminsttNm") or it.get("dmndInsttNm") or it.get("cntrctInsttNm") or it.get("insttNm") or "기관명 없음"
            mall_addr = guess_mall_addr(it)

            detail_link = it.get("cntrctDtlInfoUrl") or ""
            if not detail_link:
                unty_cntrct_no = it.get('untyCntrctNo')
                if unty_cntrct_no:
                    detail_link = f"https://www.g2b.go.kr:8067/contract/contDetail.jsp?Union_number={unty_cntrct_no}"

            base = _build_base_notice(
                "계약완료", "물품", title, client_name,
                it.get("cntrctInsttOfclTelNo", "") or it.get("telNo", ""),
                "계약 확인 필요", 0,
                it.get("cntrctAmt") or it.get("totAmt") or "",
                "확인필요",
                to_ymd(it.get("cntrctCnclsDate") or it.get("cntrctDate") or it.get("contDate")),
                detail_link
            )
            # 주소/관할 결정은 expand_and_store_with_priority에서 진행
            # → 벌크업서트를 위해 즉시 DB쓰지 말고 notice dict 자체를 모읍니다.
            # expand_and_store_with_priority 내부가 즉시 upsert/commit 구조라면,
            # '저장' 대신 '확정된 n dict'를 반환하도록 얇게 래핑해 버퍼에 추가하는 방식 권장
            n = finalize_notice_dict(base, client_code, mall_addr, client_name)  # 아래 B에서 제공
            if n: buffer.append(n)

    # 4) 벌크 업서트
    if buffer:
        bulk_upsert_notices(buffer)
        print(f"  [✅ 일괄 저장] {len(buffer)}건")





# --- 납품요구 ---


def _fetch_dlvr_detail(req_no: str):
    DLVR_DETAIL_PATH = "/1230000/at/ShoppingMallPrdctInfoService/getDlvrReqDtlInfoList"
    detail_params = {
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "inqryDiv": "2", "dlvrReqNo": req_no, "numOfRows": "100", "pageNo": "1"
    }
    try:
        data = http_get_json(api_url(DLVR_DETAIL_PATH), detail_params)
        body = _as_dict(data.get("response", {}).get("body"))
        return _as_items_list(body)
    except Exception as e:
        print(f"  [Error] 납품요구 상세 실패({req_no}): {e}")
        return []

def _fetch_dlvr_detail_with_key(req_no: str):
    return req_no, _fetch_dlvr_detail(req_no)

def fetch_and_process_delivery_requests(search_ymd: str):
    print(f"\n--- [{to_ymd(search_ymd)}] 납품요구(나라장터) 수집 ---")
    buffer = []
    CHUNK = 200  # 벌크 단위

    # 1) 총건수 1회
    first = http_get_json(api_url(DLVR_LIST_PATH), {
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": "1", "numOfRows": "1",
        "inqryDiv": "1", "inqryBgnDate": search_ymd, "inqryEndDate": search_ymd
    })
    body = _as_dict(first.get("response", {}).get("body"))
    total = int(body.get("totalCount", 0))
    if total == 0:
        print("  - 데이터 없음"); return

    page_size   = 100
    total_pages = (total + page_size - 1) // page_size
    #print(f"  - 총 {total}건 / {total_pages}")
    print(f"  - 총 {total}건")

    # 2) 페이지 병렬 (요약 목록)
    params_list = [{
        "ServiceKey": _cfg("NARA_SERVICE_KEY"), "type": "json",
        "pageNo": str(p), "numOfRows": str(page_size),
        "inqryDiv": "1", "inqryBgnDate": search_ymd, "inqryEndDate": search_ymd
    } for p in range(1, total_pages + 1)]
    pages = fetch_pages_parallel(api_url(DLVR_LIST_PATH), params_list)

    # 3) 각 페이지에서 req_no 수집 + 상세 병렬
    meta_by_req: Dict[str, Dict] = {}
    tasks = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for data in pages:
            items = _as_items_list(_as_dict(data.get("response", {}).get("body")))
            for it in items:
                req_nm = it.get("reqstNm") or it.get("dlvrReqNm") or ""
                if not is_relevant_text(req_nm):
                    continue
                req_no = it.get("dlvrReqNo") or it.get("reqstNo") or ""
                if not req_no:
                    continue

                dminstt_raw = it.get("dminsttInfo") or it.get("dmndInsttInfo") or ""
                dm_cd, dm_nm = parse_dminstt_code_from_complex(dminstt_raw)
                meta_by_req[req_no] = {
                    "req_nm": req_nm,
                    "client_code": dm_cd,
                    "client_name": dm_nm or it.get("dmndInsttNm") or it.get("dminsttNm") or "기관명 없음",
                    "mall_addr": guess_mall_addr(it),
                    "tel": it.get("cntrctDeptTelNo") or it.get("telNo") or "",
                    "rcpt": to_ymd(it.get("rcptDate") or it.get("dlvrReqRcptDate")),
                    "hdr_qty": _to_int(it.get("dlvrReqQty") or it.get("reqQty") or it.get("totQty")),
                    "hdr_amt": _to_int(it.get("dlvrReqAmt")),
                }
                tasks.append(ex.submit(_fetch_dlvr_detail_with_key, req_no))

    # 4) 상세 결과 받아서 저장(아니고: 후보 dict 수집)
    for fut in as_completed(tasks):
        req_no, products = fut.result()
        meta = meta_by_req.get(req_no)
        if not meta:
            continue

        if products:
            num_items = len(products)
            for product in products:
                prdct_nm = product.get("prdctNm") or ""
                if not is_relevant_text(meta["req_nm"], prdct_nm):
                    continue

                # 모델명 추출
                model_name = product.get("modelNm")
                if not model_name:
                    name_all = product.get("prdctIdntNoNm", "")
                    if name_all:
                        parts = [p.strip() for p in name_all.split(",")]
                        model_name = parts[2] if len(parts) >= 3 else name_all
                model_name = model_name or "모델명 없음"

                # 1) KEA API + 유사도 기반 인증 판정
                certification_status = kea_cert_with_similarity(model_name)

                # 2) 기본적으로 API 응답이 불확실하면 보수적으로 다시 체크
                if certification_status == "확인필요":
                    tmp = kea_check_certification(model_name)
                    if tmp != "확인필요":
                        certification_status = tmp

                # 수량/금액
                qty = (
                    _to_int(product.get("prdctQty"))
                    or _to_int(product.get("qty"))
                    or _to_int(product.get("orderQty"))
                    or _to_int(product.get("ordQty"))
                    or 0
                )
                if qty == 0 and num_items == 1:
                    qty = meta["hdr_qty"]

                amt_int = (
                    _to_int(product.get("prdctAmt"))
                    or _to_int(product.get("amt"))
                    or (meta["hdr_amt"] if num_items == 1 else 0)
                )
                amt = str(amt_int)  # 문자열로 저장 권장

                base = _build_base_notice(
                    "납품요구", "물품", meta["req_nm"], meta["client_name"],
                    meta["tel"], model_name, qty, amt,
                    certification_status, meta["rcpt"], f"dlvrreq:{req_no}"
                )
                n = expand_and_store_with_priority(
                    base, meta["client_code"], meta["mall_addr"], meta["client_name"], save=False
                )
                if n: buffer.append(n)
        else:
            # 상세가 비어도 헤더 한 줄 저장
            base = _build_base_notice(
                "납품요구", "물품", meta["req_nm"], meta["client_name"],
                meta["tel"], "세부내역 미확인",
                meta["hdr_qty"], str(meta["hdr_amt"]),  # 문자열로
                "확인필요", meta["rcpt"], f"dlvrreq:{req_no}"
            )
            n = expand_and_store_with_priority(
                base, meta["client_code"], meta["mall_addr"], meta["client_name"], save=False
            )
            if n: buffer.append(n)

        # 주기적 벌크 저장
        if len(buffer) >= CHUNK:
            bulk_upsert_notices(buffer); buffer.clear()

    # 남은 것 마무리
    if buffer:
        print(f"  [✅ 일괄 저장] {len(buffer)}건")
        bulk_upsert_notices(buffer)


def resolve_address_from_bjd(addr_text, bjd_code) -> str:
    """
    [주소 전용] 법정동코드→주소 보강기.
    - 절대 지사/관할 결정 함수 호출 안 함.
    - addr_text가 있으면 그걸 우선 사용.
    - 없으면 bjd_code로 bjd_mapper.get_bjd_name_str 또는 get_bjd_name 호출.
    - 항상 '문자열'을 반환(없으면 빈 문자열).
    """
    # -- 안전 문자열화
    def _as_text(x) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (int, float)):
            return str(x)
        if isinstance(x, list):
            return " ".join(_as_text(v) for v in x)
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    # 1) addr_text 우선
    addr = _as_text(addr_text).strip()

    # 2) 비었으면 bjd_code로 보강
    if not addr:
        # bjd_code를 문자열화
        code = _as_text(bjd_code).strip()
        name = ""
        if code:
            # bjd_mapper.get_bjd_name_str 우선 사용
            try:
                from bjd_mapper import get_bjd_name_str
                name = _as_text(get_bjd_name_str(code)).strip()
            except Exception:
                # 없거나 실패하면 globals의 get_bjd_name 사용
                try:
                    if "get_bjd_name" in globals():
                        name = _as_text(globals()["get_bjd_name"](code)).strip()
                except Exception:
                    name = ""
        addr = name

    # 3) 흔한 잡값 치환
    if addr in ("-", "0", "None", "null", "NULL"):
        addr = ""

    # 4) 공백 정규화
    addr = " ".join(addr.split())
    return addr




# =========================
# [GUI 연동] 구성
# =========================
# === 맨 위/설정 영역 어딘가에 추가 ===
SKIP_STAGES = {
    "order_plan": True,     # 나라장터 발주계획 스킵
    "kapt_private": True,   # K-APT 수의계약 스킵
}

# === STAGES_CONFIG 정의 바로 아래를 이처럼 바꿔주세요 ===
STAGES_CONFIG = {
    "order_plan": {"name": "발주계획(나라장터)", "func": fetch_and_process_order_plans},
    "bid_notice": {"name": "입찰공고(나라장터)", "func": fetch_and_process_bid_notices},
    "contract":   {"name": "계약완료(나라장터)", "func": fetch_and_process_contracts},
    "delivery":   {"name": "납품요구(나라장터)", "func": fetch_and_process_delivery_requests},
    "kapt_bid":   {"name": "입찰공고(K-APT)", "func": fetch_and_process_kapt_bids},
    "kapt_result":{"name": "입찰결과(K-APT)", "func": fetch_and_process_kapt_bid_results},
    "kapt_private":{"name":"수의계약(K-APT)", "func": fetch_and_process_kapt_private_contracts},
}

# ↓↓↓ 추가: 실행 대상에서 스킵값(True)인 키 제거
STAGES_CONFIG = {
    k: v for k, v in STAGES_CONFIG.items()
    if not SKIP_STAGES.get(k, False)
}


def fetch_data_for_stage(search_date: str, stage_config: dict):
    """
    gui_app.py의 SyncWorker에서 호출할 진입점.
    """
    if "func" in stage_config and isinstance(stage_config["func"], Callable):
        stage_func = stage_config["func"]
        stage_func(search_date)
    else:
        raise ValueError(f"Invalid stage_config: 'func' not found or not callable for {stage_config.get('name')}")

def get_db_session():
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def recheck_all_certifications():
    session = get_db_session()
    notices = session.query(Notice).all()

    for n in notices:
        model = n.model or n.model_name or ""
        if not model:
            n.is_certified = "확인필요"
            continue
        
        # 유사도 + KEA API 인증 확인
        cert = kea_cert_with_similarity(model)

        # 보수적 2차 체크
        if cert == "확인필요":
            tmp = kea_check_certification(model)
            if tmp != "확인필요":
                cert = tmp

        n.is_certified = cert

    session.commit()
    print("모든 기존 데이터의 인증 여부 업데이트 완료!")


if __name__ == "__main__":
    print("="*50)
    print("COLLECT_DATA.PY 단독 테스트를 시작합니다.")
    print("="*50)
    
    # 테스트할 날짜를 지정합니다. (예: '20250904')
    test_date = "20250904" 
    
    print(f"\n>>> 테스트 날짜: {test_date}\n")

    # DB 테이블을 초기화합니다. (기존 데이터 삭제)
    print("[DB] 테스트를 위해 기존 테이블을 삭제하고 새로 생성합니다.")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # 테스트하고 싶은 수집 함수를 순서대로 호출합니다.
    # 문제가 되는 K-APT 함수만 테스트하려면 아래 G2B 함수들은 주석 처리하세요.
    
    print("\n--- 나라장터(G2B) 데이터 수집 테스트 ---")
    #fetch_and_process_order_plans(test_date)
    fetch_and_process_bid_notices(test_date)
    fetch_and_process_contracts(test_date)
    fetch_and_process_delivery_requests(test_date)

    print("\n--- 공동주택(K-APT) 데이터 수집 테스트 ---")
    fetch_and_process_kapt_bids(test_date)
    #fetch_and_process_kapt_private_contracts(test_date)
    fetch_and_process_kapt_bid_results(test_date)

    print("\n>>> 테스트 완료. DB에 저장된 K-APT 데이터 건수 확인:")
    
    session = Session()
    kapt_count = session.query(Notice).filter(Notice.source_system == 'K-APT').count()
    g2b_count = session.query(Notice).filter(Notice.source_system == 'G2B').count()
    session.close()

    print(f"  - K-APT 저장 건수: {kapt_count} 건")
    print(f"  - G2B 저장 건수: {g2b_count} 건")
    print("\n[완료] `data/eers_data.db` 파일의 내용을 확인하고, GUI를 실행하여 데이터가 정상 조회되는지 확인하세요.")