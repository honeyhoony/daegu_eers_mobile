# mailer.py (SendGrid API 통합 버전)
import os
import streamlit as st
import logging
from datetime import date, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

logger = logging.getLogger(__name__)

# =======================================================
# 설정 로드
# =======================================================
try:
    import config as _local_config
except ModuleNotFoundError:
    _local_config = None

def _cfg(key, default=None):
    """환경변수 → config.py → st.secrets 순서로 탐색"""
    import os
    if key in os.environ:
        return os.environ[key]
    if _local_config and hasattr(_local_config, key):
        return getattr(_local_config, key)
    try:
        return st.secrets[key]
    except Exception:
        return default


MAIL_FROM = _cfg("MAIL_FROM", "daegu_eers@naver.com")  # Single Sender 주소
MAIL_FROM_NAME = _cfg("MAIL_FROM_NAME", "대구본부 EERS팀")
SENDGRID_API_KEY = _cfg("SENDGRID_API_KEY", "")

SIX_MONTHS = timedelta(days=180)

# =======================================================
# HTML 생성 유틸
# =======================================================
COLUMNS = [
    ("source_system", "출처"),
    ("assigned_office", "사업소"),
    ("stage", "단계"),
    ("project_name", "공고명"),
    ("client", "수요기관"),
    ("address", "주소"),
    ("phone_number", "전화"),
    ("model_name", "모델명"),
    ("quantity", "수량"),
    ("is_certified", "고효율인증"),
    ("notice_date", "공고일"),
]

def build_subject(office: str, period: Tuple[date, date], count: int) -> str:
    """메일 제목 생성"""
    start, end = period
    days = (end - start).days
    if 28 <= days <= 31:
        period_display = f"{start.month}월 전체, {count}건"
    else:
        period_display = f"{start.strftime('%m.%d')}~{end.strftime('%m.%d')}, {count}건"
    return f"[{office}] EERS 입찰공고 알림 ({period_display})"

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_rows_html(items: List[Dict]) -> str:
    trs = []
    for n in items:
        link_title = _esc(n.get("project_name") or "")
        link_url = n.get("detail_link") or ""
        link_html = f'<a href="{link_url}" target="_blank" rel="noopener">{link_title}</a>' if link_url else link_title
        tds = []
        for key, _title in COLUMNS:
            val = n.get(key)
            if key == "source_system":
                display_val = "나라장터" if str(val) == "G2B" else str(val or '')
            else:
                display_val = str(val or '')
            if key == "project_name":
                tds.append(f"<td>{link_html}</td>")
            else:
                tds.append(f"<td>{_esc(display_val)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return "\n".join(trs)

def build_table_html(items: List[Dict], for_attachment: bool = False) -> str:
    thead = "".join([f"<th>{t}</th>" for _k, t in COLUMNS])
    rows = build_rows_html(items)
    no_data_msg = '해당 년도의 누적 공고가 없습니다.' if for_attachment else '해당 기간 신규 공고가 없습니다.'
    return f"""
<table cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%;font-size:13px">
  <thead style="background:#f4f6f8"><tr>{thead}</tr></thead>
  <tbody>
    {rows if rows else f'<tr><td colspan="{len(COLUMNS)}" style="text-align:center;color:#888">{no_data_msg}</td></tr>'}
  </tbody>
</table>
"""

def build_attachment_html(office: str, year: int, items_annual: List[Dict]) -> Tuple[str, str]:
    """연간 누적 공고 HTML 첨부"""
    attach_name = f"[{office}]_{year}년_누적공고.html"
    by_month = defaultdict(list)
    for item in items_annual:
        try:
            month = int(item.get("notice_date", "0-0").split("-")[1])
            by_month[month].append(item)
        except (ValueError, IndexError):
            by_month[0].append(item)
    month_nav = [f'<a href="#month-{m}">{m}월 ({len(by_month[m])}건)</a>' for m in sorted(by_month.keys(), reverse=True)]
    monthly_tables = [f'<h3 id="month-{m}">{m}월 공고</h3>' + build_table_html(by_month[m], True) for m in sorted(by_month.keys(), reverse=True)]
    attach_html = f"""
<!doctype html><html><head><meta charset="utf-8"><title>[{office}] {year}년 누적공고</title></head>
<body><h2>[{_esc(office)}] {year}년 누적 공고</h2>
<div>{''.join(month_nav)}</div>
{''.join(monthly_tables)}
</body></html>
"""
    return attach_name, attach_html

def build_body_html(office: str, period: Tuple[date, date], items_period: List[Dict], items_annual: List[Dict]) -> Tuple[str, str, str, str]:
    """메일 본문 HTML"""
    period_txt = f"{period[0].isoformat()} ~ {period[1].isoformat()}"
    period_table = build_table_html(items_period)
    attach_name, attach_html = build_attachment_html(office, period[0].year, items_annual)
    html = f"""
<div style="font-family:Malgun Gothic,Arial,sans-serif;">
<p>안녕하세요. 대구본부 EERS팀입니다.</p>
<p><b>[{_esc(office)}]</b>의 <b>{period_txt}</b> 신규 공고 {len(items_period)}건입니다.</p>
{period_table}
<p>첨부파일에 {period[0].year}년 누적공고가 포함되어 있습니다.</p>
</div>
"""
    preview = f"[{office}] {period_txt} / {len(items_period)}건"
    return html, attach_name, attach_html, preview

# =======================================================
# SendGrid API 메일 발송
# =======================================================
def send_mail_sendgrid(
    to_list: List[str],
    subject: str,
    html_body: str,
    attach_name: str = None,
    attach_html: str = None,
):
    """SendGrid API 기반 메일 발송"""
    if not SENDGRID_API_KEY:
        st.error("⚠️ SENDGRID_API_KEY가 설정되어 있지 않습니다.")
        logger.error("SENDGRID_API_KEY missing")
        return False

    message = Mail(
        from_email=(MAIL_FROM, MAIL_FROM_NAME),
        to_emails=to_list,
        subject=subject,
        html_content=html_body,
    )

    if attach_name and attach_html:
        encoded = base64.b64encode(attach_html.encode("utf-8")).decode()
        attachment = Attachment()
        attachment.file_content = FileContent(encoded)
        attachment.file_type = FileType("text/html")
        attachment.file_name = FileName(attach_name)
        attachment.disposition = Disposition("attachment")
        message.attachment = attachment

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        if response.status_code in (200, 202):
            logger.info(f"메일 발송 성공 → {subject}")
            st.success("📨 메일이 성공적으로 발송되었습니다! (SendGrid)")
            return True
        else:
            st.error(f"메일 발송 실패: {response.status_code}")
            logger.error(f"SendGrid 응답 코드: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"[ERROR] SendGrid send_mail 실패: {e}")
        st.error(f"메일 발송 실패: {e}")
        return False

# =======================================================
# 간단한 인증코드 발송
# =======================================================
def send_verification_email(to_email: str, code: str):
    html = f"<p>[EERS 시스템]</p><p>인증코드: <b>{code}</b></p>"
    return send_mail_sendgrid(
        to_list=[to_email],
        subject="[EERS 시스템] 로그인 인증코드",
        html_body=html,
    )
