"""
Database setup — SQLAlchemy async engine + session factory.

Uses SQLite by default; swap DATABASE_URL for PostgreSQL in production.
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

from dotsocr.config import DATABASE_URL


# ── Base ───────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(20), nullable=False)  # First 8 chars for display
    name = Column(String(255), nullable=False, default="Default")
    is_active = Column(Boolean, default=True)
    rate_limit_per_minute = Column(Integer, default=10)
    rate_limit_per_day = Column(Integer, default=100)
    allowed_tasks = Column(String(255), nullable=True)  # Comma-separated: "ocr,table,formula,chart" or null=all
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # None = never expires

    # Relationships
    user = relationship("User", back_populates="api_keys")
    usage_records = relationship("UsageRecord", back_populates="api_key", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<APIKey {self.key_prefix}...>"


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    task = Column(String(50), nullable=True)  # ocr, table, formula, chart
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    elapsed_seconds = Column(Float, default=0.0)
    status_code = Column(Integer, default=200)
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_records")
    api_key = relationship("APIKey", back_populates="usage_records")

    def __repr__(self) -> str:
        return f"<UsageRecord {self.endpoint} {self.created_at}>"


# ── Engine & Session ──────────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {}
        if DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(DATABASE_URL, connect_args=connect_args)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


def init_db():
    """Create all tables and run lightweight column migrations."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _migrate_columns(engine)


def _migrate_columns(engine):
    """Add missing columns to existing tables (poor-man's migration)."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)

    # APIKey.allowed_tasks — added in v1.1
    if "api_keys" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("api_keys")]
        if "allowed_tasks" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE api_keys ADD COLUMN allowed_tasks VARCHAR(255)"))
            print("  ↳ migrated: api_keys.allowed_tasks column added")


def get_db() -> Session:
    """Dependency for FastAPI — yields a DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
