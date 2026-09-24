"""Embedding-based near-duplicate suppression (context mode, pass 2).

Deterministic, dependency-free embeddings: hashed bag-of-words vectors
(256 buckets, term frequency, L2-normalized) compared with cosine
similarity. Two texts sharing > ``threshold`` cosine are near-duplicates -
same fact, differently worded, or trivially re-cased/re-punctuated.

This is a diversity/telemetry tool, NOT a second Jev call: dedup runs
locally in code over the already-ranked candidates.
"""

from __future__ import annotations

import hashlib
import math
import re

from jev_reranker.models import RankedItem

EMBED_DIMS = 256
DEFAULT_DUP_THRESHOLD = 0.72

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def embed(text: str) -> list[float]:
    """L2-normalized hashed bag-of-words vector (deterministic across runs)."""
    vec = [0.0] * EMBED_DIMS
    for tok in _tokens(text):
        digest = hashlib.sha1(tok.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % EMBED_DIMS
        # Secondary bucket softens hash collisions.
        sign = 1.0 if digest[2] % 2 == 0 else 0.5
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def similarity(a_text: str, b_text: str) -> float:
    return cosine(embed(a_text), embed(b_text))


def suppress_near_duplicates(
    items: list[RankedItem],
    threshold: float = DEFAULT_DUP_THRESHOLD,
) -> tuple[list[RankedItem], list[RankedItem]]:
    """Split ranked items into (kept, suppressed).

    Walks items in their given order, keeping the first of any
    near-duplicate cluster; later members whose embedding cosine with an
    already-kept item is >= ``threshold`` are suppressed. A negative
    ``threshold`` disables suppression entirely. Callers wanting the
    highest-value member of each cluster kept should pass items sorted by
    descending value (``select_for_budget`` does exactly that).
    """
    if threshold < 0:
        return list(items), []
    kept: list[RankedItem] = []
    suppressed: list[RankedItem] = []
    kept_vecs: list[list[float]] = []
    for it in items:
        vec = embed(it.candidate.text)
        if any(cosine(vec, kv) >= threshold for kv in kept_vecs):
            suppressed.append(it)
        else:
            kept.append(it)
            kept_vecs.append(vec)
    return kept, suppressed
