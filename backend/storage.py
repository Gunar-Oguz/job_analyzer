"""Where the backend keeps clean jobs: PostgreSQL or plain memory.

The core works on a pandas DataFrame. A store only has to save that frame
and load it back. ``make_store()`` picks PostgreSQL when ``DATABASE_URL``
or ``POSTGRES_HOST`` is set, and an in-memory store otherwise (handy for
tests and for running the API without Docker).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pandas as pd

from core.etl import COLUMNS, LIST_COLUMNS, drop_expired, empty_frame, frame_to_records, records_to_frame


class MemoryStore:
    """Keeps the latest frame in the process. Data is lost on restart."""

    name = "memory"

    def __init__(self) -> None:
        self._frame = empty_frame()
        self._last_refresh: dict | None = None

    def init(self) -> None:
        """Nothing to prepare."""

    def save(self, frame: pd.DataFrame, report: dict) -> int:
        merged = pd.concat([self._frame, frame]).drop_duplicates("id", keep="last") if len(self._frame) else frame
        self._frame = drop_expired(merged).reset_index(drop=True)
        self._last_refresh = report
        return len(frame)

    def load(self) -> pd.DataFrame:
        return drop_expired(self._frame.copy())

    def last_refresh(self) -> dict | None:
        return self._last_refresh


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    company         TEXT,
    company_logo    TEXT,
    excerpt         TEXT,
    description     TEXT,
    employment_type TEXT,
    seniority       TEXT,
    seniority_levels TEXT[],
    category        TEXT,
    parent_categories TEXT[],
    categories      TEXT[],
    locations       TEXT[],
    is_worldwide    BOOLEAN,
    timezones       TEXT[],
    currency        TEXT,
    salary_period   TEXT,
    salary_min      DOUBLE PRECISION,
    salary_max      DOUBLE PRECISION,
    salary_usd_min  DOUBLE PRECISION,
    salary_usd_max  DOUBLE PRECISION,
    salary_usd      DOUBLE PRECISION,
    pub_date        TIMESTAMPTZ,
    expiry_date     TIMESTAMPTZ,
    apply_url       TEXT NOT NULL,
    source_url      TEXT,
    skills          TEXT[],
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_category_idx ON jobs (category);
CREATE INDEX IF NOT EXISTS jobs_expiry_idx ON jobs (expiry_date);
CREATE TABLE IF NOT EXISTS refresh_log (
    id          SERIAL PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    report      JSONB NOT NULL
);
"""


class PostgresStore:
    """Stores jobs in PostgreSQL. Every query is parametrised."""

    name = "postgres"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _dsn_from_env()

    def _connect(self):
        import psycopg2  # imported lazily so tests and the standalone app do not need it

        return psycopg2.connect(self.dsn)

    def init(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(_CREATE_SQL)

    def save(self, frame: pd.DataFrame, report: dict) -> int:
        from psycopg2.extras import execute_values

        records = frame_to_records(frame, COLUMNS)
        rows = [tuple(_db_value(column, record[column]) for column in COLUMNS) for record in records]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "id")
        sql = (
            f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES %s "
            f"ON CONFLICT (id) DO UPDATE SET {updates}, updated_at = now()"
        )
        with self._connect() as conn, conn.cursor() as cur:
            if rows:
                execute_values(cur, sql, rows, page_size=500)
            cur.execute("DELETE FROM jobs WHERE expiry_date < %s", (datetime.now(UTC),))
            cur.execute("INSERT INTO refresh_log (report) VALUES (%s)", (json.dumps(report, default=str),))
        return len(rows)

    def load(self) -> pd.DataFrame:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE expiry_date IS NULL OR expiry_date >= %s",
                (datetime.now(UTC),),
            )
            rows = cur.fetchall()
        records = [dict(zip(COLUMNS, row, strict=True)) for row in rows]
        return records_to_frame(records)

    def last_refresh(self) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT report FROM refresh_log ORDER BY fetched_at DESC LIMIT 1")
            row = cur.fetchone()
        return row[0] if row else None


def _db_value(column: str, value):
    """Lists stay lists (psycopg2 turns them into arrays); everything else passes through."""
    if column in LIST_COLUMNS:
        return [str(v) for v in (value or [])]
    return value


def _dsn_from_env() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    parts = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "job_analyzer"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }
    return " ".join(f"{key}={value}" for key, value in parts.items() if value)


def make_store():
    """PostgreSQL if it is configured, otherwise memory."""
    if os.getenv("DATABASE_URL") or os.getenv("POSTGRES_HOST"):
        return PostgresStore()
    return MemoryStore()
