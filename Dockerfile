# Python 3.10 기반 슬림 이미지
FROM python:3.10-slim

# 필수 시스템 패키지 설치 (libGL 포함)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 코드 및 파일 복사
COPY . /app

# 파이썬 패키지 설치
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Cloud Run 용 포트
EXPOSE 8080

# Streamlit 실행 명령어
CMD ["streamlit", "run", "interface.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
