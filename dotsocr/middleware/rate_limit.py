"""
Rate limiting middleware — per API key, in-memory with sliding window.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from fastapi import HTTPException, status

from dotsocr.database import APIKey


class RateLimiter:
    """Simple in-memory sliding window rate limiter."""

    def __init__(self):
        self._minute_windows: dict[int, list[float]] = defaultdict(list)
        self._day_windows: dict[int, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, api_key: APIKey) -> None:
        """
        Check if the API key is within rate limits.
        Raises HTTPException(429) if exceeded.
        """
        now = time.time()
        key_id = api_key.id

        with self._lock:
            # Clean old entries and check minute window
            minute_ago = now - 60
            self._minute_windows[key_id] = [
                t for t in self._minute_windows[key_id] if t > minute_ago
            ]
            if len(self._minute_windows[key_id]) >= api_key.rate_limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {api_key.rate_limit_per_minute} requests/minute. "
                           f"Upgrade your plan for higher limits.",
                    headers={"Retry-After": "60"},
                )

            # Check day window
            day_ago = now - 86400
            self._day_windows[key_id] = [
                t for t in self._day_windows[key_id] if t > day_ago
            ]
            if len(self._day_windows[key_id]) >= api_key.rate_limit_per_day:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Daily limit exceeded: {api_key.rate_limit_per_day} requests/day. "
                           f"Upgrade your plan for higher limits.",
                    headers={"Retry-After": "3600"},
                )

            # Record this request
            self._minute_windows[key_id].append(now)
            self._day_windows[key_id].append(now)

    def get_remaining(self, api_key: APIKey) -> dict:
        """Get remaining rate limit info."""
        now = time.time()
        key_id = api_key.id

        with self._lock:
            minute_ago = now - 60
            minute_used = len([
                t for t in self._minute_windows.get(key_id, []) if t > minute_ago
            ])
            day_ago = now - 86400
            day_used = len([
                t for t in self._day_windows.get(key_id, []) if t > day_ago
            ])

        return {
            "minute": {
                "limit": api_key.rate_limit_per_minute,
                "used": minute_used,
                "remaining": max(0, api_key.rate_limit_per_minute - minute_used),
            },
            "day": {
                "limit": api_key.rate_limit_per_day,
                "used": day_used,
                "remaining": max(0, api_key.rate_limit_per_day - day_used),
            },
        }


# Global singleton
rate_limiter = RateLimiter()
