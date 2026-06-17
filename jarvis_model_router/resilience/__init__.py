"""Resilience primitives for external service connections.

- CircuitBreaker: prevents cascading failures when Ollama is down
- HealthMonitor: periodic health checks with status tracking
- ErrorIsolator: quarantine a service after repeated errors
"""

from .circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
from .health_monitor import HealthMonitor, ServiceHealth
from .error_isolation import ErrorIsolator, ErrorCategory, QuarantinedService

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "HealthMonitor",
    "ServiceHealth",
    "ErrorIsolator",
    "ErrorCategory",
    "QuarantinedService",
]
