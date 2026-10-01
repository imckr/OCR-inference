"""
Authentication utilities — password hashing, JWT tokens, API key generation.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt

from dotsocr.config import ACCESS_TOKEN_EXPIRE_MINUTES, API_KEY_PREFIX, SECRET_KEY

# ── Password hashing ─────────────────────────────────────────────────


def hash_password(password: str) -> str:
    # bcrypt has a 72-byte limit — truncate to stay within bounds
    pw = password[:72].encode("utf-8")
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    pw = plain[:72].encode("utf-8")
    return bcrypt.checkpw(pw, hashed.encode("utf-8"))


# ── JWT Tokens ────────────────────────────────────────────────────────
ALGORITHM = "HS256"


def create_access_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ── API Key Generation ────────────────────────────────────────────────
def generate_api_key() -> str:
    """Generate a new API key like `dots_a1b2c3d4e5f6...` (48 chars)."""
    random_part = secrets.token_urlsafe(36)
    return f"{API_KEY_PREFIX}{random_part}"


def hash_api_key(key: str) -> str:
    """SHA-256 hash of the API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def get_key_prefix(key: str) -> str:
    """Get the displayable prefix of an API key (first 12 chars)."""
    return key[:12]
