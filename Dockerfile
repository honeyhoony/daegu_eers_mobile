# 1) Python 3.11 사용 (psycopg2-binary 공식 지원)
FROM python:3.11-slim AS base

# 2) 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3) 워킹 디렉토리 설정
WORKDIR /app

# 4) requirements.txt만 먼저 복사 (캐시 개선)
COPY requirements.txt .

# 5) pip 업그레이드 후 의존성 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6) 프로젝트 전체 복사
COPY . .

# 7) Streamlit 설정
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# 8) 앱 실행 (app.py 기준)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
