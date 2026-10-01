"""
Dashboard API routes — usage analytics, account stats, admin endpoints, live metrics.
"""

import logging
import time
import threading
import psutil
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from dotsocr.auth.dependencies import get_current_admin, get_current_user
from dotsocr.database import APIKey, UsageRecord, User, get_db

logger = logging.getLogger("dotsocr.dashboard")

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ══════════════════════════════════════════════════════════════════════
#  Live Metrics Collector (in-memory, watchdog-style)
# ══════════════════════════════════════════════════════════════════════
class LiveMetrics:
    """Collects real-time system and inference metrics."""
    MAX_POINTS = 120  # ~2 min at 1/sec or ~10 min at 5/sec

    def __init__(self):
        self.cpu_history = deque(maxlen=self.MAX_POINTS)
        self.memory_history = deque(maxlen=self.MAX_POINTS)
        self.rps_history = deque(maxlen=self.MAX_POINTS)  # requests per second
        self.latency_history = deque(maxlen=self.MAX_POINTS)
        self.active_requests = 0
        self.total_inferences = 0
        self.total_errors = 0
        self.uptime_start = time.time()
        self._request_times = deque(maxlen=1000)  # timestamps of recent requests
        self._latencies = deque(maxlen=100)       # recent latencies
        self._lock = threading.Lock()
        self._running = False

    def start_collector(self):
        """Start background thread collecting system metrics every 2s."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._collect_loop, daemon=True)
        t.start()

    def _collect_loop(self):
        while self._running:
            now = time.time()
            ts = datetime.utcnow().isoformat()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()

            with self._lock:
                self.cpu_history.append({"t": ts, "v": cpu})
                self.memory_history.append({"t": ts, "v": mem.percent, "used_gb": round(mem.used / (1024**3), 2), "total_gb": round(mem.total / (1024**3), 2)})

                # Calculate RPS over last 10 seconds
                cutoff = now - 10
                recent = [t for t in self._request_times if t > cutoff]
                rps = len(recent) / 10.0
                self.rps_history.append({"t": ts, "v": round(rps, 2)})

                # Average latency of recent requests
                avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0
                self.latency_history.append({"t": ts, "v": round(avg_lat, 3)})

            time.sleep(2)

    def record_request(self, latency: float, is_error: bool = False):
        """Record an inference request for live metrics."""
        with self._lock:
            self._request_times.append(time.time())
            self._latencies.append(latency)
            self.total_inferences += 1
            if is_error:
                self.total_errors += 1

    def get_snapshot(self):
        """Get current metrics snapshot."""
        with self._lock:
            now = time.time()
            cutoff_10s = now - 10
            cutoff_60s = now - 60
            recent_10 = [t for t in self._request_times if t > cutoff_10s]
            recent_60 = [t for t in self._request_times if t > cutoff_60s]
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0
            p95_lat = sorted(self._latencies)[int(len(self._latencies) * 0.95)] if len(self._latencies) > 1 else avg_lat

            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": round(now - self.uptime_start),
                "system": {
                    "cpu_percent": cpu,
                    "memory_percent": mem.percent,
                    "memory_used_gb": round(mem.used / (1024**3), 2),
                    "memory_total_gb": round(mem.total / (1024**3), 2),
                },
                "inference": {
                    "active_requests": self.active_requests,
                    "rps_10s": round(len(recent_10) / 10.0, 2),
                    "rpm_60s": len(recent_60),
                    "total_inferences": self.total_inferences,
                    "total_errors": self.total_errors,
                    "error_rate": round(self.total_errors / max(self.total_inferences, 1) * 100, 2),
                    "avg_latency": round(avg_lat, 3),
                    "p95_latency": round(p95_lat, 3),
                },
                "history": {
                    "cpu": list(self.cpu_history),
                    "memory": list(self.memory_history),
                    "rps": list(self.rps_history),
                    "latency": list(self.latency_history),
                },
            }


# Global instance
live_metrics = LiveMetrics()


# ══════════════════════════════════════════════════════════════════════
#  Usage Analytics
# ══════════════════════════════════════════════════════════════════════
@router.get("/stats")
def get_stats(
    days: int = Query(default=30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get usage statistics for the current user."""
    since = datetime.utcnow() - timedelta(days=days)

    # Total requests
    total_requests = db.query(func.count(UsageRecord.id)).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
    ).scalar() or 0

    # Total tokens
    total_tokens = db.query(func.sum(UsageRecord.total_tokens)).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
    ).scalar() or 0

    # Average latency
    avg_latency = db.query(func.avg(UsageRecord.elapsed_seconds)).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
        UsageRecord.status_code == 200,
    ).scalar() or 0.0

    # Error rate
    error_count = db.query(func.count(UsageRecord.id)).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
        UsageRecord.status_code >= 400,
    ).scalar() or 0
    error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0.0

    # Active API keys
    active_keys = db.query(func.count(APIKey.id)).filter(
        APIKey.user_id == user.id,
        APIKey.is_active == True,
    ).scalar() or 0

    # Requests by task
    task_breakdown = db.query(
        UsageRecord.task,
        func.count(UsageRecord.id),
    ).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
        UsageRecord.task != None,
    ).group_by(UsageRecord.task).all()

    # Requests by endpoint
    endpoint_breakdown = db.query(
        UsageRecord.endpoint,
        func.count(UsageRecord.id),
    ).filter(
        UsageRecord.user_id == user.id,
        UsageRecord.created_at >= since,
    ).group_by(UsageRecord.endpoint).all()

    # Daily usage for chart
    daily_usage = []
    for i in range(min(days, 30)):
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        day_count = db.query(func.count(UsageRecord.id)).filter(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= day_start,
            UsageRecord.created_at < day_end,
        ).scalar() or 0
        day_tokens = db.query(func.sum(UsageRecord.total_tokens)).filter(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= day_start,
            UsageRecord.created_at < day_end,
        ).scalar() or 0
        daily_usage.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "requests": day_count,
            "tokens": day_tokens,
        })

    daily_usage.reverse()

    return {
        "period_days": days,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "avg_latency_seconds": round(avg_latency, 2),
        "error_rate_percent": round(error_rate, 2),
        "active_api_keys": active_keys,
        "task_breakdown": {t: c for t, c in task_breakdown},
        "endpoint_breakdown": {e: c for e, c in endpoint_breakdown},
        "daily_usage": daily_usage,
    }


@router.get("/usage")
def get_usage_history(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get paginated usage history."""
    offset = (page - 1) * per_page

    total = db.query(func.count(UsageRecord.id)).filter(
        UsageRecord.user_id == user.id,
    ).scalar() or 0

    records = db.query(UsageRecord).filter(
        UsageRecord.user_id == user.id,
    ).order_by(UsageRecord.created_at.desc()).offset(offset).limit(per_page).all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "records": [
            {
                "id": r.id,
                "endpoint": r.endpoint,
                "task": r.task,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "elapsed_seconds": r.elapsed_seconds,
                "status_code": r.status_code,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }


# ══════════════════════════════════════════════════════════════════════
#  Admin Endpoints
# ══════════════════════════════════════════════════════════════════════
@router.get("/admin/users")
def admin_list_users(
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "is_active": u.is_active,
            "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat(),
            "api_keys_count": len(u.api_keys),
            "total_requests": sum(k.total_requests or 0 for k in u.api_keys),
        }
        for u in users
    ]


@router.get("/admin/stats")
def admin_stats(
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Platform-wide statistics (admin only)."""
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_keys = db.query(func.count(APIKey.id)).filter(APIKey.is_active == True).scalar() or 0
    total_requests = db.query(func.count(UsageRecord.id)).scalar() or 0
    total_tokens = db.query(func.sum(UsageRecord.total_tokens)).scalar() or 0

    # Last 24h
    day_ago = datetime.utcnow() - timedelta(days=1)
    requests_24h = db.query(func.count(UsageRecord.id)).filter(
        UsageRecord.created_at >= day_ago,
    ).scalar() or 0

    return {
        "total_users": total_users,
        "total_active_keys": total_keys,
        "total_requests": total_requests,
        "total_tokens": total_tokens or 0,
        "requests_last_24h": requests_24h,
    }


# ══════════════════════════════════════════════════════════════════════
#  Live Metrics Endpoint (polling)
# ══════════════════════════════════════════════════════════════════════
@router.get("/live")
def get_live_metrics(
    user: User = Depends(get_current_user),
):
    """Get real-time system and inference metrics for live dashboard."""
    return live_metrics.get_snapshot()


@router.get("/processors")
def get_available_processors(
    user: User = Depends(get_current_user),
):
    """Get available processing tasks/backends."""
    from dotsocr.inference.engine import get_backend, get_device, get_model_info
    from dotsocr.config import VALID_TASKS, TASK_PROMPTS

    backend = get_backend()
    device = get_device()

    return {
        "backend": backend,
        "device": device,
        "available_tasks": [
            {"id": t, "name": t.replace("_", " ").title(), "prompt": TASK_PROMPTS[t]}
            for t in VALID_TASKS
            if t in TASK_PROMPTS
        ],
    }
