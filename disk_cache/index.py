"""Small SQLite index used as the authoritative LRU ledger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Generator

from .errors import IndexVersionError


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    namespace: str
    source_path: str
    source_dev: int
    source_ino: int
    source_size: int
    source_mtime_ns: int
    cache_suffix: str
    cached_size: int
    created_ns: int
    last_access_ns: int


class CacheIndex:
    """Thread-safe-by-construction index using one connection per operation."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise IndexVersionError(
                    f"Unsupported cache index version {version}; "
                    f"expected {SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    source_path TEXT NOT NULL UNIQUE,
                    source_dev INTEGER NOT NULL,
                    source_ino INTEGER NOT NULL,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    cache_suffix TEXT NOT NULL,
                    cached_size INTEGER NOT NULL CHECK (cached_size >= 0),
                    created_ns INTEGER NOT NULL,
                    last_access_ns INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS entries_lru ON entries(last_access_ns, key)"
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def get(self, key: str) -> CacheEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM entries WHERE key = ?", (key,)
            ).fetchone()
        return _entry(row) if row is not None else None

    def upsert(self, entry: CacheEntry) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM entries WHERE source_path = ? AND key <> ?",
                (entry.source_path, entry.key),
            )
            connection.execute(
                """
                INSERT INTO entries (
                    key, namespace, source_path, source_dev, source_ino,
                    source_size, source_mtime_ns, cache_suffix, cached_size,
                    created_ns, last_access_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    namespace = excluded.namespace,
                    source_path = excluded.source_path,
                    source_dev = excluded.source_dev,
                    source_ino = excluded.source_ino,
                    source_size = excluded.source_size,
                    source_mtime_ns = excluded.source_mtime_ns,
                    cache_suffix = excluded.cache_suffix,
                    cached_size = excluded.cached_size,
                    created_ns = excluded.created_ns,
                    last_access_ns = excluded.last_access_ns
                """,
                (
                    entry.key,
                    entry.namespace,
                    entry.source_path,
                    entry.source_dev,
                    entry.source_ino,
                    entry.source_size,
                    entry.source_mtime_ns,
                    entry.cache_suffix,
                    entry.cached_size,
                    entry.created_ns,
                    entry.last_access_ns,
                ),
            )

    def touch(self, key: str, last_access_ns: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE entries SET last_access_ns = ? WHERE key = ?",
                (last_access_ns, key),
            )

    def delete(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM entries WHERE key = ?", (key,))

    def entries(self) -> list[CacheEntry]:
        return self._select("SELECT * FROM entries ORDER BY key")

    def lru_entries(self, excluded_key: str | None = None) -> list[CacheEntry]:
        if excluded_key is None:
            return self._select("SELECT * FROM entries ORDER BY last_access_ns, key")
        return self._select(
            "SELECT * FROM entries WHERE key <> ? ORDER BY last_access_ns, key",
            (excluded_key,),
        )

    def total_size(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(SUM(cached_size), 0) FROM entries"
                ).fetchone()[0]
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def _select(
        self, statement: str, parameters: tuple[object, ...] = ()
    ) -> list[CacheEntry]:
        with self._connect() as connection:
            rows = connection.execute(statement, parameters).fetchall()
        return [_entry(row) for row in rows]

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _entry(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(**dict(row))
