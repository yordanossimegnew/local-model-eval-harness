"""PostgreSQL storage: runs, one row per (model, suite, example) result, per-model resource stats."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, delete, func, insert, select, update)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.engine import Engine

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("name", Text),
    Column("status", String(12), nullable=False),        # queued | running | completed | failed
    Column("config", JSONB, nullable=False),             # models, suites, limit, json_mode override
    Column("progress", JSONB, nullable=False, default=dict),
    Column("summary", JSONB),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
)
results = Table(
    "results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("model", Text, nullable=False),               # "<runtime>/<model>"
    Column("suite", Text, nullable=False),
    Column("example_id", Text, nullable=False),
    Column("prompt", Text, nullable=False),
    Column("output", Text, nullable=False, default=""),
    Column("tool_calls", JSONB, nullable=False, default=list),
    Column("expected", JSONB),
    Column("score", Float, nullable=False, default=0.0),
    Column("passed", Boolean, nullable=False, default=False),
    Column("detail", Text),
    Column("json_valid", Boolean),
    Column("tool_valid", Boolean),
    Column("latency_ms", Float),
    Column("ttft_ms", Float),
    Column("output_tokens", Integer),
    Column("tokens_estimated", Boolean, default=False),
    Column("tokens_per_sec", Float),
    Column("error", Text),
    UniqueConstraint("run_id", "model", "suite", "example_id", name="uq_result"),
)
model_stats = Table(
    "model_stats", metadata,
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
    Column("model", Text, primary_key=True),
    Column("load_ms", Float),
    Column("resources", JSONB, nullable=False, default=dict),  # runtime-reported memory + host CPU/RAM samples
    Column("error", Text),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _row(r) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r._mapping.items()}


class Store:
    def __init__(self, database_url: str):
        url = database_url
        for p in ("postgresql://", "postgres://"):
            if url.startswith(p):
                url = "postgresql+psycopg://" + url[len(p):]
        self.engine: Engine = create_engine(url, pool_pre_ping=True)

    def init(self) -> None:
        metadata.create_all(self.engine)

    def create_run(self, config: dict, name: str | None = None) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.engine.begin() as c:
            c.execute(insert(runs).values(id=rid, name=name, status="queued", config=config, progress={},
                                          created_at=now()))
        return rid

    def update_run(self, rid: str, **values) -> None:
        if values.get("status") in ("completed", "failed"):
            values.setdefault("finished_at", now())
        with self.engine.begin() as c:
            c.execute(update(runs).where(runs.c.id == rid).values(**values))

    def run(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(runs).where(runs.c.id == rid)).first()
        return _row(r) if r else None

    def runs(self, limit: int = 50) -> list[dict]:
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(select(runs).order_by(runs.c.created_at.desc()).limit(limit))]

    def delete_run(self, rid: str) -> bool:
        with self.engine.begin() as c:
            return bool(c.execute(delete(runs).where(runs.c.id == rid)).rowcount)

    def fail_interrupted(self) -> None:
        with self.engine.begin() as c:
            c.execute(update(runs).where(runs.c.status.in_(["running", "queued"]))
                      .values(status="failed", error="interrupted (process restarted); resume it", finished_at=now()))

    def save_result(self, rid: str, **values) -> None:
        stmt = pg_insert(results).values(run_id=rid, **values)
        with self.engine.begin() as c:
            c.execute(stmt.on_conflict_do_update(constraint="uq_result", set_=values))

    def done_keys(self, rid: str) -> set[tuple[str, str, str]]:
        with self.engine.connect() as c:
            return {(r.model, r.suite, r.example_id) for r in
                    c.execute(select(results.c.model, results.c.suite, results.c.example_id)
                              .where(results.c.run_id == rid, results.c.error.is_(None)))}

    def count(self, rid: str) -> int:
        with self.engine.connect() as c:
            return c.execute(select(func.count()).select_from(results).where(results.c.run_id == rid)).scalar()

    def results(self, rid: str, model: str | None = None, suite: str | None = None) -> list[dict]:
        q = select(results).where(results.c.run_id == rid).order_by(results.c.model, results.c.suite, results.c.id)
        if model:
            q = q.where(results.c.model == model)
        if suite:
            q = q.where(results.c.suite == suite)
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(q)]

    def save_model_stats(self, rid: str, model: str, **values) -> None:
        stmt = pg_insert(model_stats).values(run_id=rid, model=model, **values)
        with self.engine.begin() as c:
            c.execute(stmt.on_conflict_do_update(index_elements=["run_id", "model"], set_=values))

    def model_stats(self, rid: str) -> dict[str, dict]:
        with self.engine.connect() as c:
            return {r.model: _row(r) for r in c.execute(select(model_stats).where(model_stats.c.run_id == rid))}
