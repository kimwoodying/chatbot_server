# 🚀 chatbot_server - Django 챗봇 서버 (VM 배포)

**배포 방식**: GCP VM 직접 배포  
**서버 주소**: 34.42.223.43  
**Django 포트**: 8001  
**외부 공개 포트**: 80/443, 8001

---

## 🎯 빠른 배포 (6단계)

```bash
# 1. 파일 업로드 (로컬 → VM)
scp -r chatbot_server ubuntu@34.42.223.43:/home/ubuntu/

# 2. SSH 접속
ssh ubuntu@34.42.223.43

# 3. 가상환경 및 패키지 설치
cd /home/ubuntu/chatbot_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 환경변수 (.env)
# .env 파일을 /home/ubuntu/chatbot_server 에 위치시키고
# FAISS_INDEX_PATH=/home/ubuntu/chatbot_server/chatbot/data/faiss.index
# METADATA_PATH=/home/ubuntu/chatbot_server/chatbot/data/metadata.json
# 인지 꼭 확인

# 5. Django 설정
python manage.py migrate
python manage.py collectstatic --noinput

# 6. 서버 실행
gunicorn -w 4 -b 0.0.0.0:8001 chat_django.wsgi:application
```

---

## ⚠️ 오류 가능 지점 (OS/서버 차이)

- Windows ↔ Ubuntu: `faiss-cpu`는 Ubuntu에서 대체로 정상 설치되지만, Windows에서는 휠 부재로 설치 실패할 수 있음.
- `torch==2.9.1`: Ubuntu에서도 Python 버전/아키텍처에 따라 휠이 없으면 빌드 이슈가 생길 수 있음.
- `channels` 사용 시: 설치만으로 끝나지 않으며, WebSocket을 실제로 쓰면 `ASGI_APPLICATION`, `CHANNEL_LAYERS` 설정과 ASGI 서버(daphne/uvicorn)가 필요함.
- `channels`를 import만 하고 WebSocket을 쓰지 않으면 `gunicorn` + WSGI로도 당장은 동작함.
        WSGI(gunicorn wsgi): HTTP 요청/응답만 처리 (일반 REST API/페이지 OK)
        ASGI(daphne/uvicorn asgi): HTTP + WebSocket 둘 다 처리 (실시간 채팅/푸시 필요)

---

## 🔐 환경 변수 (.env)

`/home/user/project/chatbot_server/.env`에 생성

```env
# Django
DEBUG=False
SECRET_KEY=change-me
ALLOWED_HOSTS=34.42.223.43
TOOL_AUTH_REQUIRED=true
CORS_ALLOW_ALL_ORIGINS=false
CORS_ALLOWED_ORIGINS=http://34.42.223.43,https://34.42.223.43
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=true
SECURE_HSTS_PRELOAD=true
SECURE_SSL_REDIRECT=true
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true
CACHE_CLEAR_ENABLED=true
CACHE_CLEAR_HOUR=4
CACHE_CLEAR_MINUTE=0

# LLM / RAG
PRIMARY_LLM=openai
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini
GROQ_API_KEY=your-groq-key
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
FAISS_INDEX_PATH=/home/user/project/chatbot_server/chatbot/data/faiss.index
METADATA_PATH=/home/user/project/chatbot_server/chatbot/data/metadata.json

# External APIs
HOLIDAY_API_KEY=your-holiday-key

# Default DB (MySQL, 병원 DB로 사용 시)
USE_SQLITE=false
MYSQL_HOST=34.42.223.43
MYSQL_PORT=3306
MYSQL_DATABASE=hospital_db
MYSQL_USER=acorn
MYSQL_PASSWORD=acorn1234

# Hospital DB (hospital alias, tool 조회용)
HOSPITAL_DATABASE_URL=mysql://acorn:acorn1234@34.42.223.43:3306/hospital_db
HOSPITAL_RESERVATION_TABLE=patients_appointment
``` 

- FAISS/METADATA 경로는 **VM 실제 경로 기준으로 수정**. 로컬(Windows) 경로는 사용하지 않음.
- 해당 파일이 없다면 `python manage.py ingest_documents`로 생성.
- 기본 DB를 병원 DB로 사용 시 `USE_SQLITE=false` 및 `MYSQL_HOST/MYSQL_PORT/MYSQL_DATABASE/MYSQL_USER/MYSQL_PASSWORD` 설정 필요.
- hospital alias도 쓰려면 `HOSPITAL_DATABASE_URL`을 기본 DB와 동일하게 맞추는 방식이 안전.

---

## 🏗️ 프로젝트 구조

```
GCP VM (34.42.223.43)
│
├── Nginx (80/443)
├── FastAPI (8000)
├── Django 챗봇 (8001)  ← chatbot_server
├── AI Models (5001)
├── Qdrant (6333)
└── MySQL (3306)
```

---

## 📦 포함 파일

- manage.py
- requirements.txt
- chat_django/
- chatbot/ (migrations, services)
- static/ (127개)
- .env (운영 시, 별도 전달)

**총 192개 파일** (15-20 MB)

---

## 🔓 방화벽/보안그룹

- 외부 공개: 80/443 (Nginx), 8001 (Django 직결)
- 내부 전용: 8000, 5001, 6333, 3306

---

## 🔄 Systemd 자동 실행

`/etc/systemd/system/django-chatbot.service`:
```ini
[Unit]
Description=Django Chatbot
After=network.target

[Service]
User=your-user
WorkingDirectory=/home/user/project/chatbot_server
EnvironmentFile=/home/user/project/chatbot_server/.env
Environment="PATH=/home/user/project/chatbot_server/venv/bin"
ExecStart=/home/user/project/chatbot_server/venv/bin/gunicorn -w 4 -b 0.0.0.0:8001 chat_django.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl start django-chatbot
sudo systemctl enable django-chatbot
```

## Cache clear (daily)

Built-in scheduler clears cache once per day while the app is running.
Override with `CACHE_CLEAR_ENABLED`, `CACHE_CLEAR_HOUR`, `CACHE_CLEAR_MINUTE`.

Optional: use `manage.py clear_chat_cache` via systemd timer.

```bash
sudo cp ops/systemd/django-chatbot-cache-clear.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now django-chatbot-cache-clear.timer
```

---

## ⚙️ Nginx 설정

```nginx
upstream django_chatbot {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name 34.42.223.43;

    location /api/chat/ {
        proxy_pass http://django_chatbot;
        proxy_set_header Host $host;
    }

    location /admin/ {
        proxy_pass http://django_chatbot;
    }

    location /static/ {
        alias /home/user/project/chatbot_server/static/;
    }
}
```

8001 포트를 외부 공개하므로, Nginx 없이도 직접 접근 가능. 80/443 사용 시 위 설정 유지.

---

## 📊 포트

| 서비스 | 포트 | 외부 공개 |
|--------|------|-----------|
| Nginx | 80/443 | O |
| FastAPI | 8000 | X |
| **Django** | **8001** | O |
| AI Models | 5001 | X |
| Qdrant | 6333 | X |
| MySQL | 3306 | X |

---

## 🧪 API 테스트

```bash
# 1) Django 직결 (8001 외부 공개)
curl -X POST http://34.42.223.43:8001/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message":"병원 전화번호"}'

# 2) Nginx 경유 (80)
curl -X POST http://34.42.223.43/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message":"병원 전화번호"}'
```

---

**생성일**: 2026-01-12 10:44  
**배포**: GCP VM  
**포트**: 8001

**VM 배포 준비 완료! 🚀**
