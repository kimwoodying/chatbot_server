import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

def _get_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def _get_csv(name, default):
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_int(name, default=0):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


SECRET_KEY = os.getenv("SECRET_KEY", "replace-this-with-a-secure-secret-key")
DEBUG = _get_bool("DEBUG", False)

ALLOWED_HOSTS = _get_csv("ALLOWED_HOSTS", ["127.0.0.1", "localhost"])  # GCP 외부 IP 지정 가능

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "channels",
    "corsheaders",
    "chatbot.apps.ChatbotConfig",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "chat_django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "chat_django.wsgi.application"
ASGI_APPLICATION = "chat_django.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}


def _parse_mysql_url(url: str) -> dict:
    parsed = urlparse(url)
    name = (parsed.path or "").lstrip("/")
    return {
        "ENGINE": "django.db.backends.mysql",
        "NAME": name,
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": parsed.hostname or "",
        "PORT": str(parsed.port or 3306),
        "OPTIONS": {"charset": "utf8mb4"},
    }


# Database configuration - supports both SQLite (for dev/test) and MySQL (for production)
USE_SQLITE = _get_bool("USE_SQLITE", True)
MYSQL_NAME = os.getenv("MYSQL_DATABASE", "hospital_db")
MYSQL_USER = os.getenv("MYSQL_USER", "acorn")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "acorn1234")
MYSQL_HOST = os.getenv("MYSQL_HOST", "34.42.223.43")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")

if USE_SQLITE:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    # Remote MySQL Database Configuration
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": MYSQL_NAME,
            "USER": MYSQL_USER,
            "PASSWORD": MYSQL_PASSWORD,
            "HOST": MYSQL_HOST,  # Remote MySQL server
            "PORT": MYSQL_PORT,
            "OPTIONS": {
                "charset": "utf8mb4",
                "connect_timeout": 10,
            },
        }
    }

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.db.DatabaseCache",
        "LOCATION": "django_cache",
    }
}

hospital_db_url = os.getenv("HOSPITAL_DATABASE_URL") or os.getenv("DATABASE_URL")
if hospital_db_url:
    DATABASES["hospital"] = _parse_mysql_url(hospital_db_url)

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "static"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# CORS (Flutter 에뮬레이터 & React 개발용)
CORS_ALLOW_ALL_ORIGINS = _get_bool("CORS_ALLOW_ALL_ORIGINS", False)
CORS_ALLOWED_ORIGINS = _get_csv("CORS_ALLOWED_ORIGINS", [])
CORS_ALLOW_CREDENTIALS = False
CORS_ALLOW_HEADERS = ["*"]
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]

SECURE_HSTS_SECONDS = _get_int("SECURE_HSTS_SECONDS", 0)
SECURE_HSTS_INCLUDE_SUBDOMAINS = _get_bool("SECURE_HSTS_INCLUDE_SUBDOMAINS", False)
SECURE_HSTS_PRELOAD = _get_bool("SECURE_HSTS_PRELOAD", False)
SECURE_SSL_REDIRECT = _get_bool("SECURE_SSL_REDIRECT", False)
SESSION_COOKIE_SECURE = _get_bool("SESSION_COOKIE_SECURE", False)
CSRF_COOKIE_SECURE = _get_bool("CSRF_COOKIE_SECURE", False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise RuntimeError("GROQ_API_KEY 환경변수가 설정되어 있지 않습니다.")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional: only required when Gemini is enabled.

# App logging: show chatbot/service info logs in console.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file_info": {
            "class": "logging.FileHandler",
            "filename": str(BASE_DIR / "runserver-8001.log"),
            "formatter": "verbose",
            "level": "INFO",
            "encoding": "utf-8",
        },
        "file_error": {
            "class": "logging.FileHandler",
            "filename": str(BASE_DIR / "runserver-8001.err.log"),
            "formatter": "verbose",
            "level": "ERROR",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "chatbot": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False,
        },
        "chatbot.services": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "file_error"],
        "level": "WARNING",
    },
}
