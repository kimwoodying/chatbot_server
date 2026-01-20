# 🚀 chatbot_server
### Django 기반 병원 챗봇 서버

> **Django + LLM(RAG)** 기반 챗봇 서버로, 병원 안내, 예약 조회, 증상 기반 응답 등을 제공합니다.  
> 내부 시스템과 연동되는 구조로 설계되었으며, REST API를 통해 접근 가능합니다.

---

# 📌 서비스 개요

- **Backend**: Django (Gunicorn)
- **LLM**: OpenAI / Groq
- **Vector DB**: FAISS
- **Database**: MySQL
- **API**: REST API
- **배포**: Google Cloud Platform (VM)

---

# 🛠 기술 스택 선택 이유

- **Django**: 인증, 관리자 페이지, ORM 활용에 적합
- **Gunicorn**: 안정적인 WSGI 서버 구성
- **FAISS**: 빠른 벡터 검색이 필요한 병원 문서 RAG에 적합
- **MySQL**: 병원 기존 시스템과의 호환성
- **GCP VM**: 안정적인 클라우드 배포 환경

---

# 📋 요구사항

- Python 3.8+
- Django 3.2+
- MySQL 5.7+
- pip (패키지 관리)

---

# 🚀 설치 방법

### 1. 저장소 복제
```bash
git clone https://github.com/yourusername/chatbot_server.git
cd chatbot_server
```

### 2. 가상 환경 생성 및 활성화
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 프로젝트 루트에 생성하고 다음과 같이 설정하세요:
```env
DEBUG=False
SECRET_KEY=your-secret-key-here
DATABASE_URL=mysql://user:password@localhost:3306/chatbot_db
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 5. 데이터베이스 마이그레이션
```bash
python manage.py migrate
python manage.py createsuperuser
```

### 6. 개발 서버 실행
```bash
python manage.py runserver
```

서버는 `http://localhost:8000`에서 실행됩니다.

---

# 🔄 요청 처리 흐름

1. 사용자가 챗봇 API로 메시지 전송
2. 의도 분류 (정보 조회 / 증상 문의 / 예약 관련)
3. 필요 시 DB 또는 내부 API 조회
4. FAISS 기반 문서 검색(RAG)
5. LLM을 통한 응답 생성
6. 결과 캐싱 후 사용자에게 응답

---

# 📚 API 사용 예시

### 챗봇 메시지 전송
```bash
POST /api/chat/message/
Content-Type: application/json

{
  "message": "진료과를 알려주세요"
}
```

---

# 👥 기여 가이드라인

버그 리포트와 기능 요청은 [Issues](../../issues)를 통해 제출해주세요.  
Pull Request도 환영합니다!

### 개발 환경 설정
```bash
pip install -r requirements-dev.txt
pytest  # 테스트 실행
```

---

# 📄 라이센스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

---


