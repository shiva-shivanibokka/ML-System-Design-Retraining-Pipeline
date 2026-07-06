"""Lightweight in-memory rate limiting for the public serving endpoints.

Free-tier friendly: no external store or dependency. A per-client sliding
window bounds abuse of the endpoints that do expensive work — model inference
(/predict) and an outbound LLM call (/drift/explain). Single-process only
(the HF Space runs one worker), which is exactly this deployment.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class SlidingWindowLimiter:
    """Allow at most `max_requests` per `window_seconds` per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        dq = self._hits[key]
        cutoff = now - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True


def make_rate_limiter(max_requests: int, window_seconds: float):
    """Return a FastAPI dependency that enforces the given per-client limit."""
    limiter = SlidingWindowLimiter(max_requests, window_seconds)

    def dependency(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({max_requests}/{int(window_seconds)}s). Slow down.",
            )

    return dependency
