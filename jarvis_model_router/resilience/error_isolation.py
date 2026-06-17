"""Error Isolation -- categorize, track, and quarantine failing services.

Inspired by code_puppy's MCPErrorIsolator pattern. When a service
starts erroring in a specific category, it gets quarantined so that
subsequent calls fail fast instead of waiting for timeout/exhaustion.

Error categories:
  TIMEOUT    -- provider didn't respond in time
  CONNECTION -- network-level failure (DNS, refused, etc.)
  HTTP_4XX   -- client errors (401, 403, 429, etc.)
  HTTP_5XX   -- server errors (500, 502, 503, etc.)
  PARSE      -- response couldn't be parsed
  UNKNOWN    -- uncategorized
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    PARSE = "parse"
    UNKNOWN = "unknown"


@dataclass
class ErrorStats:
    """Error statistics for a service."""

    category: ErrorCategory
    count: int = 0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    last_message: str = ""


@dataclass
class QuarantinedService:
    """A service that has been quarantined due to errors."""

    service_name: str
    category: ErrorCategory
    reason: str
    quarantined_at: float
    release_at: float
    error_count: int

    @property
    def is_expired(self) -> bool:
        """Check if the quarantine period has elapsed."""
        return time.monotonic() >= self.release_at


class ErrorIsolator:
    """Isolates failing services by category with quarantine support.

    Args:
        quarantine_duration_ms: How long to quarantine a service
        quarantine_threshold: Errors before quarantine kicks in
    """

    def __init__(
        self,
        quarantine_duration_ms: int = 60_000,
        quarantine_threshold: int = 5,
    ) -> None:
        self._quarantine_duration_ms = quarantine_duration_ms
        self._quarantine_threshold = quarantine_threshold

        self._error_stats: Dict[str, Dict[ErrorCategory, ErrorStats]] = defaultdict(
            dict
        )
        self._quarantined: Dict[str, QuarantinedService] = {}
        self._lock = threading.Lock()

    def record_error(
        self,
        service_name: str,
        category: ErrorCategory,
        message: str = "",
    ) -> None:
        """Record an error for a service."""
        now = time.monotonic()

        with self._lock:
            stats = self._error_stats[service_name].get(category)
            if stats is None:
                stats = ErrorStats(category=category, first_seen=now)
                self._error_stats[service_name][category] = stats

            stats.count += 1
            stats.last_seen = now
            stats.last_message = message

            # Check quarantine threshold
            if stats.count >= self._quarantine_threshold:
                self._quarantine(service_name, category, message)

    def _quarantine(
        self,
        service_name: str,
        category: ErrorCategory,
        reason: str,
    ) -> None:
        """Quarantine a service."""
        stats = self._error_stats[service_name].get(category)
        count = stats.count if stats else 0

        now = time.monotonic()
        release_at = now + (self._quarantine_duration_ms / 1000.0)

        self._quarantined[service_name] = QuarantinedService(
            service_name=service_name,
            category=category,
            reason=reason,
            quarantined_at=now,
            release_at=release_at,
            error_count=count,
        )

        logger.warning(
            "Quarantined service '%s' (category=%s, errors=%d, duration=%dms)",
            service_name,
            category.value,
            count,
            self._quarantine_duration_ms,
        )

    def is_quarantined(self, service_name: str) -> bool:
        """Check if a service is currently quarantined."""
        with self._lock:
            qs = self._quarantined.get(service_name)
            if qs is None:
                return False
            if qs.is_expired:
                # Auto-release expired quarantines
                del self._quarantined[service_name]
                logger.info("Quarantine expired for '%s', releasing", service_name)
                return False
            return True

    def get_quarantine_info(self, service_name: str) -> Optional[QuarantinedService]:
        """Return quarantine details for a service, or None."""
        with self._lock:
            qs = self._quarantined.get(service_name)
            if qs and not qs.is_expired:
                return qs
            return None

    def release(self, service_name: str) -> bool:
        """Manually release a quarantined service."""
        with self._lock:
            if service_name in self._quarantined:
                del self._quarantined[service_name]
                logger.info("Manually released quarantine for '%s'", service_name)
                return True
            return False

    def get_stats(self, service_name: str) -> Dict[str, ErrorStats]:
        """Return error statistics for a service by category."""
        with self._lock:
            return {
                cat.value: stats
                for cat, stats in self._error_stats.get(service_name, {}).items()
            }

    def get_all_quarantined(self) -> List[QuarantinedService]:
        """Return all currently quarantined services."""
        with self._lock:
            # Prune expired
            expired = [name for name, qs in self._quarantined.items() if qs.is_expired]
            for name in expired:
                del self._quarantined[name]

            return list(self._quarantined.values())

    def clear(self, service_name: Optional[str] = None) -> None:
        """Clear error stats and quarantines."""
        with self._lock:
            if service_name:
                self._error_stats.pop(service_name, None)
                self._quarantined.pop(service_name, None)
            else:
                self._error_stats.clear()
                self._quarantined.clear()


def categorize_http_status(status_code: int) -> ErrorCategory:
    """Categorize an HTTP status code into an ErrorCategory."""
    if 400 <= status_code < 500:
        return ErrorCategory.HTTP_4XX
    if 500 <= status_code < 600:
        return ErrorCategory.HTTP_5XX
    return ErrorCategory.UNKNOWN
