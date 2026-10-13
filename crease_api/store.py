"""Save checks to PostgreSQL, and read summaries back.

Optional: with no DATABASE_URL set (local runs, tests) every function is
a no-op, so the checker never depends on the database being up. A failed
save is logged and swallowed for the same reason.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("crease.store")
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
_ready = False


def enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _connect():
    import psycopg   # imported lazily: the API runs without it
    return psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=5, autocommit=True)


def _ensure(conn) -> None:
    global _ready
    if not _ready:
        # Read only when a database is actually used, so a missing file can
        # never take down the rest of the API.
        conn.execute(SCHEMA_FILE.read_text())
        _ready = True


def save(source: str, report: dict, pattern: dict) -> int | None:
    if not enabled():
        return None
    try:
        with _connect() as conn:
            _ensure(conn)
            row = conn.execute(
                """insert into checks (source, creases, vertices, flat_foldable,
                                       failing_vertices, rules, elapsed_ms, pattern)
                   values (%s, %s, %s, %s, %s, %s, %s, %s) returning id""",
                (source, report["checked"]["creases"], report["checked"]["vertices"],
                 report["flat_foldable"], report["failing_vertices"],
                 [v["rule"] for v in report["violations"]], report["elapsed_ms"],
                 json.dumps(pattern)),
            ).fetchone()
            return row[0]
    except Exception as e:   # never let storage break a check
        log.warning("could not save check: %s", e)
        return None


def recent(limit: int = 10) -> list[dict]:
    if not enabled():
        return []
    with _connect() as conn:
        _ensure(conn)
        rows = conn.execute(
            """select id, created_at, source, creases, vertices, flat_foldable, failing_vertices
               from checks order by created_at desc limit %s""", (limit,)).fetchall()
    keys = ["id", "created_at", "source", "creases", "vertices", "flat_foldable", "failing_vertices"]
    return [dict(zip(keys, r)) | {"created_at": r[1].isoformat()} for r in rows]


def stats() -> dict:
    """How patterns fail, across everything ever checked."""
    if not enabled():
        return {}
    with _connect() as conn:
        _ensure(conn)
        total, passed, photos = conn.execute(
            """select count(*), count(*) filter (where flat_foldable),
                      count(*) filter (where source = 'photo') from checks""").fetchone()
        by_rule = conn.execute(
            """select rule, count(*) as times, count(distinct id) as patterns
               from checks, unnest(rules) as rule
               group by rule order by times desc""").fetchall()
        median_ms = conn.execute(
            "select percentile_cont(0.5) within group (order by elapsed_ms) from checks").fetchone()[0]
    return {
        "checks": total,
        "passed": passed,
        "from_photos": photos,
        "median_ms": median_ms,
        "rules_broken": [{"rule": r, "times": t, "patterns": p} for r, t, p in by_rule],
    }
