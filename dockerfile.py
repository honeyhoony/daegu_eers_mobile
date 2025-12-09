# ─────────────────────────────────────
# 1) Python 베이스 이미지
# ─────────────────────────────────────
FROM python:3.13-slim

# 시스템 패키지 (psycopg2용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────
# 2) 작업 디렉토리
# ─────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────
# 3) 파이썬 패키지 설치
# ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────
# 4) 앱 소스 복사
# ─────────────────────────────────────
COPY . .

# Streamlit 포트
ENV PORT=8501

EXPOSE 8501

# ─────────────────────────────────────
# 5) Streamlit 실행
# ─────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
