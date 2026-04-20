"""
In-Memory Session Store (Thread-Safe, TTL-Based)
=================================================
Maintains conversational history for multi-turn interactions.  Each session
stores up to ``SESSION_MAX_TURNS`` previous query/response turns, which are
injected as additional context into the LLM prompt.  This enables coherent
follow-up questions without the need for a database.

Design decisions
----------------
* Pure Python standard library — no Redis, no Postgres, no extra dependencies.
* Thread-safe via ``threading.Lock``.
* Expired sessions are lazily evicted on each ``get_session`` call so no
  background thread is needed.
* Session IDs are 128-bit random hex strings (UUID4) with negligible collision
  probability.

Limitations (acknowledged for research transparency)
-----------------------------------------------------
* State is lost on process restart (in-process only).
* Not suitable for multi-worker deployments without a shared cache layer.
  For production use, replace the ``_sessions`` dict with a Redis TTL store.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

# Session time-to-live: 30 minutes of inactivity
SESSION_TTL_SECONDS: int = 1800

# Maximum number of turns retained per session
SESSION_MAX_TURNS: int = 5


@dataclass
class Turn:
    """A single query/response exchange within a session."""

    query: str
    symptoms: list[str]
    conditions: list[str]   # possible_conditions from response
    explanation: str        # brief explanation for context
    severity: str           # low | medium | high


@dataclass
class Session:
    """A conversational session tracking multiple turns."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > SESSION_MAX_TURNS:
            self.turns.pop(0)
        self.last_active = time.time()

    def context_summary(self) -> str:
        """
        Build a compact textual summary of previous turns for LLM injection.

        Returns an empty string if no prior turns exist (first query in session).
        """
        if not self.turns:
            return ""
        lines = ["[Prior conversation context — same patient session:]"]
        for i, t in enumerate(self.turns, 1):
            sym_str = ", ".join(t.symptoms) if t.symptoms else "not specified"
            cond_str = ", ".join(t.conditions) if t.conditions else "none identified"
            lines.append(
                f"  Turn {i}: Query='{t.query}'; "
                f"Symptoms=[{sym_str}]; "
                f"Conditions considered=[{cond_str}]; "
                f"Severity={t.severity}."
            )
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0


class SessionStore:
    """
    Thread-safe in-memory session store with lazy TTL eviction.

    Usage
    -----
    Instantiate once at application startup and store in ``app.state``::

        store = SessionStore()
        app.state.session_store = store

    Then in route handlers::

        sid, session = app.state.session_store.get_or_create(payload.session_id)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        sid = uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = Session(session_id=sid)
        return sid

    def get_session(self, session_id: str) -> Session | None:
        """Return the session for *session_id*, or ``None`` if expired/missing."""
        self._evict_expired()
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        """
        Return ``(session_id, session)``.

        If *session_id* is ``None`` or not found (expired), a fresh session is
        created and its ID is returned alongside the new ``Session`` object.
        """
        if session_id:
            session = self.get_session(session_id)
            if session is not None:
                return session_id, session
        new_id = self.create_session()
        with self._lock:
            return new_id, self._sessions[new_id]

    def record_turn(
        self,
        session_id: str,
        query: str,
        symptoms: list[str],
        result: dict,
    ) -> None:
        """Append a completed turn to the session history (no-op if session gone)."""
        session = self.get_session(session_id)
        if session is None:
            return
        session.add_turn(
            Turn(
                query=query,
                symptoms=symptoms,
                conditions=result.get("possible_conditions", []),
                explanation=result.get("explanation", "")[:200],
                severity=result.get("severity", "low"),
            )
        )

    def __len__(self) -> int:
        """Return number of active (non-expired) sessions."""
        self._evict_expired()
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        with self._lock:
            expired = [k for k, v in self._sessions.items() if v.last_active < cutoff]
            for k in expired:
                del self._sessions[k]
