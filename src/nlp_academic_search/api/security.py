"""Small single-instance security controls for the public research API."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Fixed-window-compatible sliding limiter for one Render worker."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def subject(token: str, client_host: str) -> str:
        value = token or client_host or "unknown"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def allow(self, subject: str, bucket: str, limit: int, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - 60.0
        key = (subject, bucket)
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(current)
            return True
