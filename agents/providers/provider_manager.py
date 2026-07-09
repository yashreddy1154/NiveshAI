"""
NiveshAI — Provider Manager
Factory pattern for creating LLM providers + usage tracking.
"""
# TODO: Implement in Phase 2

import sqlite3
from datetime import datetime
from pathlib import Path


class UsageTracker:
    """Tracks API usage per provider in SQLite."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "usage.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tokens_input INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()

    def log_usage(self, provider: str, model: str, tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO api_usage (provider, model, tokens_input, tokens_output, cost_usd) VALUES (?, ?, ?, ?, ?)",
            (provider, model, tokens_in, tokens_out, cost),
        )
        conn.commit()
        conn.close()

    def get_daily_usage(self, provider: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        today = datetime.now().strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_input + tokens_output), 0), COALESCE(SUM(cost_usd), 0) "
            "FROM api_usage WHERE provider=? AND DATE(timestamp)=?",
            (provider, today),
        ).fetchone()
        conn.close()
        return {"requests": row[0], "tokens": row[1], "cost": row[2]}


class ProviderManager:
    """Factory for creating and managing LLM providers."""

    def __init__(self):
        self.usage_tracker = UsageTracker()
        self._active_provider = None

    def get_provider(self, provider_name: str, api_key: str = None):
        """Create and return a provider instance."""
        raise NotImplementedError("Will be implemented in Phase 2")

    def set_active(self, provider_name: str, api_key: str = None):
        """Set the active LLM provider."""
        raise NotImplementedError("Will be implemented in Phase 2")

    @property
    def active(self):
        return self._active_provider
