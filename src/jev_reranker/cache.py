"""Two-level cache (in-memory + JSON file) for Jev judgments.

Cache key = sha256(query | candidate ids+texts | model | policy version |
question-schema version | judged heads | rubric id). Any coefficient/schema/
mode-head/rubric change invalidates the cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from jev_reranker.rubric import DEFAULT_RUBRIC_ID


def cache_key(
    query: str,
    candidate_texts: list[tuple[str, str]],
    model: str,
    policy_version: str,
    schema_version: str,
    heads: tuple[str, ...] = ("rel", "util", "sup", "con"),
    rubric_id: str = DEFAULT_RUBRIC_ID,
) -> str:
    payload = json.dumps(
        {
            "q": query,
            "c": [[cid, text] for cid, text in candidate_texts],
            "m": model,
            "p": policy_version,
            "s": schema_version,
            "h": list(heads),
            "r": rubric_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgmentCache:
    """Thread-safe memory cache with optional JSON-file persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self._mem: dict[str, Any] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            try:
                self._mem = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._mem = {}

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._mem[key] = value
            if self._path is not None:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._path.write_text(json.dumps(self._mem), encoding="utf-8")
                except OSError:
                    pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._mem)
