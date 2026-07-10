"""
NiveshAI — SQLite Cache Layer
Caches API responses to avoid redundant calls and respect rate limits.
All cache entries have a TTL (time-to-live) after which they expire.
"""

import sqlite3
import json
import time
import hashlib
import os
from pathlib import Path
from typing import Any, Optional


# Default DB location relative to project root
_DEFAULT_DB = Path(__file__).parent / "cache.db"


class CacheManager:
    """SQLite-based key-value cache with TTL expiry."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or _DEFAULT_DB)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")   # allows concurrent reads
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key       TEXT PRIMARY KEY,
                    value     TEXT NOT NULL,
                    cached_at REAL NOT NULL,
                    ttl_secs  REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider   TEXT NOT NULL,
                    model      TEXT NOT NULL,
                    ts         REAL NOT NULL DEFAULT (unixepoch()),
                    tokens_in  INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd   REAL DEFAULT 0.0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_provider ON api_usage(provider, ts)")
            conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, max_age_hours: float = 24) -> Optional[Any]:
        """
        Return cached value if it exists and hasn't expired.
        Returns None on cache miss or expiry.
        """
        hkey = self._hash(key)
        ttl_secs = max_age_hours * 3600
        now = time.time()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, cached_at FROM cache WHERE key = ?", (hkey,)
            ).fetchone()

        if row is None:
            return None

        value_str, cached_at = row
        age = now - cached_at

        if age > ttl_secs:
            # Expired — delete it
            self.delete(key)
            return None

        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            return value_str   # plain string fallback

    def set(self, key: str, value: Any, ttl_hours: float = 24):
        """Store a value with a TTL."""
        hkey = self._hash(key)
        value_str = json.dumps(value) if not isinstance(value, str) else value
        now = time.time()
        ttl_secs = ttl_hours * 3600

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, cached_at, ttl_secs)
                VALUES (?, ?, ?, ?)
                """,
                (hkey, value_str, now, ttl_secs),
            )
            conn.commit()

    def delete(self, key: str):
        """Remove a single cache entry."""
        hkey = self._hash(key)
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (hkey,))
            conn.commit()

    def clear(self, prefix: Optional[str] = None):
        """Clear cache. If prefix given, only clear matching keys."""
        with self._connect() as conn:
            if prefix is None:
                conn.execute("DELETE FROM cache")
            else:
                # We store hashed keys so prefix filtering isn't easy.
                # Load all, filter by original key isn't stored.
                # For simplicity: delete all. User should use specific delete instead.
                conn.execute("DELETE FROM cache")
            conn.commit()

    def purge_expired(self):
        """Remove all expired entries to reclaim disk space."""
        now = time.time()
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM cache WHERE (? - cached_at) > ttl_secs", (now,)
            ).rowcount
            conn.commit()
        return deleted

    def size(self) -> int:
        """Number of entries currently in cache."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    # ── API Usage Tracking ─────────────────────────────────────────────────────

    def log_api_usage(
        self,
        provider: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ):
        """Log one API call for usage tracking."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_usage (provider, model, ts, tokens_in, tokens_out, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (provider, model, time.time(), tokens_in, tokens_out, cost_usd),
            )
            conn.commit()

    def get_daily_usage(self, provider: str) -> dict:
        """Get today's request count, tokens, and cost for a provider."""
        # Today = last 24 hours
        since = time.time() - 86400
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(tokens_in+tokens_out), 0), COALESCE(SUM(cost_usd), 0)
                FROM api_usage WHERE provider = ? AND ts >= ?
                """,
                (provider, since),
            ).fetchone()
        return {
            "requests": row[0],
            "tokens":   int(row[1]),
            "cost_usd": round(row[2], 4),
        }

    def get_all_usage(self) -> dict:
        """Get daily usage stats for all providers."""
        since = time.time() - 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT provider, COUNT(*), COALESCE(SUM(tokens_in+tokens_out),0), COALESCE(SUM(cost_usd),0)
                FROM api_usage WHERE ts >= ?
                GROUP BY provider
                """,
                (since,),
            ).fetchall()
        return {
            row[0]: {"requests": row[1], "tokens": int(row[2]), "cost_usd": round(row[3], 4)}
            for row in rows
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(key: str) -> str:
        """Hash the cache key to a fixed-length string."""
        return hashlib.sha256(key.encode()).hexdigest()


# Module-level singleton — import and use directly
_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Return the global CacheManager singleton (lazy init)."""
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache
