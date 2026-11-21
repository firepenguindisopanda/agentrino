"""Server-issued admin sessions and login throttling.

Login exchanges the admin password for a random, expiring bearer token, so the
password is sent once instead of on every request and is never stored by the
browser. Tokens are opaque (no claims to forge) and only their SHA-256 digest is
kept, so a memory dump of this store cannot be replayed.

State is in-process: a restart (or a second instance) signs admins out, which
is acceptable for a single-instance admin console. Move to Redis if the API is
ever scaled horizontally.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

DEFAULT_TTL_SECONDS = 8 * 60 * 60
MAX_SESSIONS = 100  # oldest sessions are evicted beyond this

LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 5 * 60
_MAX_TRACKED_CLIENTS = 1000


def ttl_seconds() -> int:
    """ADMIN_SESSION_TTL_HOURS from the environment (default 8, clamped to 5 min..7 days)."""
    raw = os.getenv("ADMIN_SESSION_TTL_HOURS", "")
    try:
        hours = float(raw) if raw else DEFAULT_TTL_SECONDS / 3600
    except ValueError:
        hours = DEFAULT_TTL_SECONDS / 3600
    return int(min(max(hours * 3600, 300), 7 * 24 * 3600))


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class Session:
    token: str
    expires_at: float  # epoch seconds


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._sessions: OrderedDict[str, float] = OrderedDict()  # digest -> expires_at
        self._max = max_sessions

    def create(self, ttl: int | None = None) -> Session:
        self._purge()
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + (ttl if ttl is not None else ttl_seconds())
        self._sessions[_digest(token)] = expires_at
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return Session(token=token, expires_at=expires_at)

    def expires_at(self, token: str | None) -> float | None:
        """Expiry of a live session, or None if the token is unknown or expired."""
        if not token:
            return None
        key = _digest(token)
        expires_at = self._sessions.get(key)
        if expires_at is None:
            return None
        if expires_at <= time.time():
            del self._sessions[key]
            return None
        return expires_at

    def revoke(self, token: str | None) -> None:
        if token:
            self._sessions.pop(_digest(token), None)

    def clear(self) -> None:
        self._sessions.clear()

    def _purge(self) -> None:
        now = time.time()
        for key in [k for k, exp in self._sessions.items() if exp <= now]:
            del self._sessions[key]


class LoginThrottle:
    """Sliding-window limit on failed logins per client key (IP address)."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window: int = LOGIN_WINDOW_SECONDS):
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._max = max_failures
        self._window = window

    def _recent(self, key: str) -> deque[float]:
        attempts = self._failures.get(key)
        if attempts is None:
            return deque()
        cutoff = time.time() - self._window
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if not attempts:
            del self._failures[key]
        return attempts

    def retry_after(self, key: str) -> int:
        """Seconds until another attempt is allowed (0 = allowed now)."""
        attempts = self._recent(key)
        if len(attempts) < self._max:
            return 0
        return max(1, int(attempts[0] + self._window - time.time()) + 1)

    def record_failure(self, key: str) -> None:
        attempts = self._recent(key)
        attempts.append(time.time())
        self._failures[key] = attempts
        self._failures.move_to_end(key)
        while len(self._failures) > _MAX_TRACKED_CLIENTS:
            self._failures.popitem(last=False)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._failures.clear()
        else:
            self._failures.pop(key, None)


sessions = SessionStore()
login_throttle = LoginThrottle()
