"""
Auth API routes — registration, login, API key management.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from dotsocr.auth.dependencies import get_current_user
from dotsocr.auth.utils import (
    create_access_token,
    generate_api_key,
    get_key_prefix,
    hash_api_key,
    hash_password,
    verify_password,
)
from dotsocr.config import FREE_TIER_REQUESTS_PER_DAY, FREE_TIER_REQUESTS_PER_MINUTE
from dotsocr.database import APIKey, User, get_db

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(default="Default", max_length=255)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)
    allowed_tasks: Optional[list[str]] = Field(default=None, description="Tasks this key can access. Null = all tasks.")


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    is_active: bool
    rate_limit_per_minute: int
    rate_limit_per_day: int
    allowed_tasks: Optional[list[str]]
    total_requests: int
    total_tokens: int
    last_used_at: Optional[str]
    created_at: str
    expires_at: Optional[str]


class APIKeyCreatedResponse(APIKeyResponse):
    """Returned only on creation — includes the full key (shown once)."""
    key: str


# ── Registration ──────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=req.email,
        name=req.name,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "name": user.name},
    )


# ── Login ─────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "name": user.name},
    )


# ── Profile ───────────────────────────────────────────────────────────
@router.get("/me")
def get_profile(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat(),
    }


# ── API Key Management ───────────────────────────────────────────────
@router.post("/api-keys", response_model=APIKeyCreatedResponse, status_code=201)
def create_api_key(
    req: CreateAPIKeyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Limit to 5 active keys per user
    active_count = db.query(APIKey).filter(
        APIKey.user_id == user.id,
        APIKey.is_active == True,
    ).count()
    if active_count >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 active API keys per account.",
        )

    raw_key = generate_api_key()
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)

    api_key = APIKey(
        user_id=user.id,
        key_hash=hash_api_key(raw_key),
        key_prefix=get_key_prefix(raw_key),
        name=req.name,
        rate_limit_per_minute=FREE_TIER_REQUESTS_PER_MINUTE,
        rate_limit_per_day=FREE_TIER_REQUESTS_PER_DAY,
        allowed_tasks=','.join(req.allowed_tasks) if req.allowed_tasks else None,
        expires_at=expires_at,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return APIKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        rate_limit_per_day=api_key.rate_limit_per_day,
        allowed_tasks=api_key.allowed_tasks.split(',') if api_key.allowed_tasks else None,
        total_requests=api_key.total_requests,
        total_tokens=api_key.total_tokens,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@router.get("/api-keys", response_model=list[APIKeyResponse])
def list_api_keys(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    keys = db.query(APIKey).filter(APIKey.user_id == user.id).order_by(APIKey.created_at.desc()).all()
    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            is_active=k.is_active,
            rate_limit_per_minute=k.rate_limit_per_minute,
            rate_limit_per_day=k.rate_limit_per_day,
            allowed_tasks=k.allowed_tasks.split(',') if k.allowed_tasks else None,
            total_requests=k.total_requests,
            total_tokens=k.total_tokens,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == user.id,
    ).first()
    if api_key is None:
        raise HTTPException(status_code=404, detail="API key not found.")

    api_key.is_active = False
    db.commit()
    return None
