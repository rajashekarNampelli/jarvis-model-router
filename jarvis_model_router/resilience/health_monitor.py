"""Health Monitor -- periodic health checks with status tracking.

Monitors the health of providers (Ollama, external APIs) and
maintains a rolling window of health status for observability.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health record for a single service."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: Optional[float] = None
    last_healthy_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    latency_ms: Optional[int] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "last_check": self.last_check_time,
        }


class HealthMonitor:
    """Periodic health monitor for registered services.

    Usage:
        monitor = HealthMonitor(check_interval_seconds=30)
        monitor.register("ollama", ollama_health_check)
        await monitor.start()
        # ... later ...
        health = monitor.get_health("ollama")
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        failure_threshold: int = 3,
        success_threshold: int = 2,
    ) -> None:
        self._check_interval = check_interval_seconds
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold

        self._checks: Dict[str, Callable[..., Coroutine[Any, Any, bool]]] = {}
        self._health: Dict[str, ServiceHealth] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    def register(
        self,
        service_name: str,
        check_fn: Callable[..., Coroutine[Any, Any, bool]],
    ) -> None:
        """Register a health check function for a service."""
        self._checks[service_name] = check_fn
        self._health[service_name] = ServiceHealth(name=service_name)
        logger.debug("Registered health check for: %s", service_name)

    def unregister(self, service_name: str) -> None:
        """Remove a service from health monitoring."""
        self._checks.pop(service_name, None)
        self._health.pop(service_name, None)

    async def start(self) -> None:
        """Start the periodic health check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            "Health monitor started (interval=%ds, services=%d)",
            self._check_interval,
            len(self._checks),
        )

    async def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _check_loop(self) -> None:
        """Background loop that periodically checks all services."""
        while self._running:
            for name, check_fn in list(self._checks.items()):
                try:
                    await self._perform_check(name, check_fn)
                except Exception:
                    logger.debug("Health check error for %s", name, exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _perform_check(
        self,
        name: str,
        check_fn: Callable[..., Coroutine[Any, Any, bool]],
    ) -> None:
        """Execute a single health check and update status."""
        health = self._health.get(name)
        if health is None:
            return

        start = time.monotonic()
        try:
            is_healthy = await check_fn()
            latency_ms = int((time.monotonic() - start) * 1000)

            health.last_check_time = time.monotonic()
            health.total_checks += 1
            health.latency_ms = latency_ms

            if is_healthy:
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.last_healthy_time = time.monotonic()

                if health.status != HealthStatus.HEALTHY:
                    if health.consecutive_successes >= self._success_threshold:
                        health.status = HealthStatus.HEALTHY
                        health.message = "Service is healthy"
            else:
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.total_failures += 1

                if health.consecutive_failures >= self._failure_threshold:
                    health.status = HealthStatus.UNHEALTHY
                    health.message = (
                        f"Unhealthy after {health.consecutive_failures} failures"
                    )
                else:
                    health.status = HealthStatus.DEGRADED
                    health.message = (
                        f"Degraded ({health.consecutive_failures} failures)"
                    )

        except asyncio.TimeoutError:
            health.consecutive_failures += 1
            health.total_failures += 1
            health.status = HealthStatus.UNHEALTHY
            health.message = "Health check timed out"
        except Exception as exc:
            health.consecutive_failures += 1
            health.total_failures += 1
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Health check error: {exc}"

    async def check_now(self, service_name: str) -> ServiceHealth:
        """Force an immediate health check for a service."""
        check_fn = self._checks.get(service_name)
        if check_fn is None:
            raise KeyError(f"No health check registered for '{service_name}'")
        await self._perform_check(service_name, check_fn)
        return self._health[service_name]

    def get_health(self, service_name: str) -> ServiceHealth:
        """Return the current health status for a service."""
        return self._health.get(service_name, ServiceHealth(name=service_name))

    def get_all_health(self) -> Dict[str, ServiceHealth]:
        """Return health status for all registered services."""
        return self._health.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict for API exposure."""
        return {name: health.to_dict() for name, health in self._health.items()}
