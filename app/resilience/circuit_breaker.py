"""Circuit Breaker -- prevent cascading failures from failing providers.

States: CLOSED (healthy) -> OPEN (failing, reject calls) -> HALF_OPEN (probing).

When a provider exceeds the failure threshold, the circuit opens and
all subsequent calls are rejected immediately (no waiting for timeout).
After a cooldown period, the circuit enters HALF_OPEN and allows one
probe call. If it succeeds, the circuit closes; if it fails, it reopens.

Inspired by code_puppy's CircuitBreaker implementation.
"""

import logging
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted against an open circuit."""

    def __init__(self, service_name: str, remaining_ms: int) -> None:
        self.service_name = service_name
        self.remaining_ms = remaining_ms
        super().__init__(
            f"Circuit is OPEN for '{service_name}' (retry in {remaining_ms}ms)"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for a single service.

    Args:
        service_name: Identifier for logging
        failure_threshold: Consecutive failures before opening
        recovery_timeout_ms: Milliseconds before attempting half-open probe
        success_threshold: Consecutive successes in half-open before closing
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 30000,
        success_threshold: int = 2,
    ) -> None:
        self.service_name = service_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_ms = recovery_timeout_ms
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition on read)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN if cooldown has elapsed."""
        if self._last_failure_time is None:
            return

        elapsed_ms = int((time.monotonic() - self._last_failure_time) * 1000)
        if elapsed_ms >= self._recovery_timeout_ms:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            logger.info(
                "Circuit '%s': OPEN -> HALF_OPEN (after %dms cooldown)",
                self.service_name,
                elapsed_ms,
            )

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns True if the circuit is CLOSED or HALF_OPEN (probe).
        Raises CircuitOpenError if the circuit is OPEN.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                # Allow one probe at a time (simplified)
                return True
            # OPEN -- check cooldown
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.HALF_OPEN:
                return True

            elapsed_ms = (
                int((time.monotonic() - self._last_failure_time) * 1000)
                if self._last_failure_time
                else 0
            )
            remaining = self._recovery_timeout_ms - elapsed_ms
            raise CircuitOpenError(self.service_name, max(0, remaining))

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        "Circuit '%s': HALF_OPEN -> CLOSED (probe succeeded)",
                        self.service_name,
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed -- reopen
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s': HALF_OPEN -> OPEN (probe failed)",
                    self.service_name,
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit '%s': CLOSED -> OPEN (%d consecutive failures)",
                        self.service_name,
                        self._failure_count,
                    )

    def reset(self) -> None:
        """Force the circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("Circuit '%s': force-reset to CLOSED", self.service_name)

    def get_stats(self) -> dict:
        """Return current circuit statistics."""
        with self._lock:
            return {
                "service": self.service_name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
            }
