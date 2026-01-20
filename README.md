🚀 chatbot_server — Django 챗봇 서버 (GCP VM)

GCP VM에 직접 배포한 운영용 Django 챗봇 서버

배포 방식: GCP VM 수동 배포 (systemd)

서버 IP: 34.42.223.43

Django 포트: 8001

외부 공개 포트: 80 / 443 / 8001

📌 전체 아키텍처
GCP VM (34.42.223.43)
│
├── Nginx (80 / 443)
├── Django 챗봇 서버 (8001)   ← chatbot_server
├── FastAPI (8000)            (내부 통신)
├── AI 모델 서버 (5001)       (내부 통신)
├── Qdrant (6333)             (내부 통신)
└── MySQL (3306)              (내부 통신)

⚡ 빠른 배포 (6단계)
# 1. 프로젝트 업로드 (로컬 → VM)
scp -r chatbot_server ubuntu@34.42.223.43:/home/ubuntu/

# 2. VM 접속
ssh ubuntu@34.42.223.43

# 3. 가상환경 생성 및 패키지 설치
cd /home/ubuntu/chatbot_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 환경변수 설정 (.env)
# /home/ubuntu/chatbot_server/.env 위치에 생성

# 5. Django 초기 설정
python manage.py migrate
python manage.py collectstatic --noinput

# 6. 서버 실행
gunicorn -w 4 -b 0.0.0.0:8001 chat_django.wsgi:application

🔐 환경 변수 (.env)

📍 위치

/home/ubuntu/chatbot_server/.env

# =====================
# Django 기본 설정
# =====================
DEBUG=False
SECRET_KEY=change-me
ALLOWED_HOSTS=34.42.223.43

# 보안 설정
SECURE_SSL_REDIRECT=true
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=true
SECURE_HSTS_PRELOAD=true

# CORS
CORS_ALLOW_ALL_ORIGINS=false
CORS_ALLOWED_ORIGINS=http://34.42.223.43,https://34.42.223.43

# =====================
# 캐시 설정
# =====================
CACHE_CLEAR_ENABLED=true
CACHE_CLEAR_HOUR=4
CACHE_CLEAR_MINUTE=0

# =====================
# LLM / RAG
# =====================
PRIMARY_LLM=openai
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=gpt-4o-mini

GROQ_API_KEY=YOUR_GROQ_KEY
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask

FAISS_INDEX_PATH=/home/ubuntu/chatbot_server/chatbot/data/faiss.index
METADATA_PATH=/home/ubuntu/chatbot_server/chatbot/data/metadata.json

# =====================
# 외부 API
# =====================
HOLIDAY_API_KEY=YOUR_HOLIDAY_API_KEY

# =====================
# DB 설정
# =====================
USE_SQLITE=false

MYSQL_HOST=34.42.223.43
MYSQL_PORT=3306
MYSQL_DATABASE=hospital_db
MYSQL_USER=acorn
MYSQL_PASSWORD=YOUR_DB_PASSWORD

# 병원 DB alias (툴 조회용)
HOSPITAL_DATABASE_URL=mysql://acorn:YOUR_DB_PASSWORD@34.42.223.43:3306/hospital_db
HOSPITAL_RESERVATION_TABLE=patients_appointment
