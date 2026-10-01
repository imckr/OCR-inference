"""
FastAPI authentication dependencies.
"""

from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from dotsocr.auth.utils import decode_access_token, hash_api_key
from dotsocr.database import APIKey, User, get_db

# ── Bearer token security scheme ──────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Extracts and validates the JWT from the Authorization header.
    Used for dashboard/account endpoints.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Require admin privileges."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def _extract_api_key(request: Request) -> Optional[str]:
    """
    Extract API key from:
    1. Authorization: Bearer dots_xxx header
    2. X-API-Key header
    3. api_key query parameter
    """
    # Check Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer dots_"):
        return auth_header[7:]  # Remove "Bearer "

    # Check X-API-Key header
    api_key = request.headers.get("x-api-key")
    if api_key and api_key.startswith("dots_"):
        return api_key

    # Check query parameter
    api_key = request.query_params.get("api_key")
    if api_key and api_key.startswith("dots_"):
        return api_key

    return None


async def validate_api_key(
    request: Request,
    db: Session = Depends(get_db),
) -> tuple[User, APIKey]:
    """
    Validates an API key for inference endpoints.
    Returns (user, api_key) tuple.
    """
    raw_key = _extract_api_key(request)

    if raw_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Get one at the DotsOCR dashboard.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_hash = hash_api_key(raw_key)
    api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True,
    ).first()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired.",
        )

    user = db.query(User).filter(User.id == api_key.user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account associated with this API key is inactive.",
        )

    # Update last used
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    return user, api_key
