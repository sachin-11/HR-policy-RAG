"""Semantic cache for HR policy Q&A.

Instead of exact-match caching, we embed each query and compare against
cached queries using cosine similarity.  If a similar-enough query was
answered before, we return the cached answer immediately — skipping the
full RAG pipeline and LLM call.

Storage: JSON file (dev). Swap for Redis + pgvector in production.

Usage:
    cache = SemanticCache()
    hit = cache.lookup(user_message, embedder)
    if hit:
        return hit["answer"]
    # ... run full pipeline ...
    cache.store(user_message, answer, embedder)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from app.observability.logging import log_event

logger = logging.getLogger("app.cache")

CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data/cache"))
CACHE_FILE = CACHE_DIR / "semantic_cache.json"
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "500"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60 * 24)))  # 24h default


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _load_cache() -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_cache(entries: list[dict[str, Any]]) -> None:
    CACHE_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


class SemanticCache:
    """In-process semantic similarity cache backed by a JSON file.

    For production replace _load_cache / _save_cache with Redis or a vector DB.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._entries: list[dict[str, Any]] = _load_cache()

    def lookup(self, query: str, embedder: Any) -> dict[str, Any] | None:
        """Return a cached entry if a semantically similar query exists, else None.

        `embedder` must have an `embed(text: str) -> list[float]` method.
        """
        if not self._entries or embedder is None:
            return None

        try:
            query_vec = embedder.embed(query)
        except Exception as exc:
            log_event(logger, event="cache.embed_error", error=str(exc))
            return None

        now = time.time()
        best_score = 0.0
        best_entry: dict[str, Any] | None = None

        for entry in self._entries:
            # Skip expired entries
            if now - entry.get("ts", 0) > CACHE_TTL_SECONDS:
                continue
            stored_vec = entry.get("embedding", [])
            if not stored_vec:
                continue
            score = _cosine_similarity(query_vec, stored_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            log_event(logger, event="cache.hit", score=round(best_score, 4), query=query[:80])
            best_entry["hits"] = best_entry.get("hits", 0) + 1
            _save_cache(self._entries)
            return best_entry

        log_event(logger, event="cache.miss", best_score=round(best_score, 4), query=query[:80])
        return None

    def store(self, query: str, answer: str, embedder: Any, sources: list | None = None) -> None:
        """Embed the query and store the answer in the cache."""
        if embedder is None:
            return
        try:
            embedding = embedder.embed(query)
        except Exception as exc:
            log_event(logger, event="cache.store_error", error=str(exc))
            return

        entry = {
            "query": query,
            "answer": answer,
            "sources": sources or [],
            "embedding": embedding,
            "ts": time.time(),
            "hits": 0,
        }
        self._entries.append(entry)

        # Evict oldest entries beyond MAX_CACHE_ENTRIES
        if len(self._entries) > MAX_CACHE_ENTRIES:
            self._entries = sorted(self._entries, key=lambda e: e.get("ts", 0))
            self._entries = self._entries[-MAX_CACHE_ENTRIES:]

        _save_cache(self._entries)
        log_event(logger, event="cache.stored", query=query[:80])

    def invalidate_all(self) -> None:
        """Clear the entire cache (e.g. after policy documents are updated)."""
        self._entries = []
        _save_cache(self._entries)
        log_event(logger, event="cache.invalidated")

    @property
    def size(self) -> int:
        return len(self._entries)


# ── module-level singleton ─────────────────────────────────────────────────────
_cache_instance: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
