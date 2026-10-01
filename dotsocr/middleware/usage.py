"""
Usage tracking — records every inference request for analytics.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from dotsocr.database import APIKey, UsageRecord

logger = logging.getLogger("dotsocr.usage")


def record_usage(
    db: Session,
    user_id: int,
    api_key_id: int,
    endpoint: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_seconds: float = 0.0,
    status_code: int = 200,
    task: Optional[str] = None,
    error_message: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> None:
    """Record a usage event and update API key counters."""
    try:
        record = UsageRecord(
            user_id=user_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            task=task,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed_seconds,
            status_code=status_code,
            error_message=error_message,
            ip_address=ip_address,
        )
        db.add(record)

        # Update API key counters
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
        if api_key:
            api_key.total_requests = (api_key.total_requests or 0) + 1
            api_key.total_tokens = (api_key.total_tokens or 0) + prompt_tokens + completion_tokens
            api_key.last_used_at = datetime.utcnow()

        db.commit()
    except Exception as exc:
        logger.error(f"Failed to record usage: {exc}", exc_info=True)
        db.rollback()
