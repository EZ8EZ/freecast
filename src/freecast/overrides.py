"""Planner override layer with a durable, queryable audit trail.

This is the accountability layer buyers of legacy demand-planning tools
actually pay for: not the statistics, but being able to answer "who changed
this number, when, and why" for every SKU-period a planner touched. Every
override is appended, never mutated in place, so the full history is always
reconstructable — including overrides that were later superseded.

Backed by a local DuckDB file so there's no separate database to stand up,
consistent with freecast's local-first, zero-ops design.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS override_id_seq START 1;

CREATE TABLE IF NOT EXISTS overrides (
    override_id BIGINT PRIMARY KEY,
    unique_id VARCHAR NOT NULL,
    ds DATE NOT NULL,
    model VARCHAR NOT NULL,
    baseline_y_hat DOUBLE NOT NULL,
    override_y_hat DOUBLE NOT NULL,
    author VARCHAR NOT NULL,
    reason VARCHAR,
    created_at TIMESTAMP NOT NULL,
    superseded_by BIGINT
);
"""


@dataclass(frozen=True)
class Override:
    override_id: int
    unique_id: str
    ds: dt.date
    model: str
    baseline_y_hat: float
    override_y_hat: float
    author: str
    reason: str | None
    created_at: dt.datetime


class OverrideStore:
    """Append-only override log backed by a local DuckDB file."""

    def __init__(self, path: str | Path = "freecast_overrides.duckdb") -> None:
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._con.execute(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> OverrideStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(
        self,
        *,
        unique_id: str,
        ds: dt.date,
        model: str,
        baseline_y_hat: float,
        override_y_hat: float,
        author: str,
        reason: str | None = None,
    ) -> Override:
        """Record a new override, superseding any prior active override for
        the same (unique_id, ds)."""
        if not author:
            raise ValueError("author is required — overrides must be attributable.")

        now = dt.datetime.now(dt.timezone.utc)
        seq_row = self._con.execute("SELECT nextval('override_id_seq')").fetchone()
        assert seq_row is not None
        override_id = seq_row[0]

        self._con.execute(
            """
            UPDATE overrides
            SET superseded_by = ?
            WHERE unique_id = ? AND ds = ? AND superseded_by IS NULL
            """,
            [override_id, unique_id, ds],
        )
        self._con.execute(
            """
            INSERT INTO overrides
                (override_id, unique_id, ds, model, baseline_y_hat, override_y_hat,
                 author, reason, created_at, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            [
                override_id,
                unique_id,
                ds,
                model,
                baseline_y_hat,
                override_y_hat,
                author,
                reason,
                now,
            ],
        )
        return Override(
            override_id=override_id,
            unique_id=unique_id,
            ds=ds,
            model=model,
            baseline_y_hat=baseline_y_hat,
            override_y_hat=override_y_hat,
            author=author,
            reason=reason,
            created_at=now,
        )

    def active_overrides(self) -> pl.DataFrame:
        """The single current override per (unique_id, ds) — supersession chains collapsed."""
        return self._con.execute(
            "SELECT * FROM overrides WHERE superseded_by IS NULL ORDER BY unique_id, ds"
        ).pl()

    def audit_log(self, *, unique_id: str | None = None, ds: dt.date | None = None) -> pl.DataFrame:
        """Full history for a series/date — every version, superseded or not."""
        query = "SELECT * FROM overrides WHERE 1=1"
        params: list[object] = []
        if unique_id is not None:
            query += " AND unique_id = ?"
            params.append(unique_id)
        if ds is not None:
            query += " AND ds = ?"
            params.append(ds)
        query += " ORDER BY unique_id, ds, created_at"
        return self._con.execute(query, params).pl()

    def apply(self, forecasts: pl.DataFrame) -> pl.DataFrame:
        """Left-join active overrides onto a freecast forecasts frame.

        Adds ``y_hat_final`` (the override value where one exists, else the
        statistical ``y_hat``) and ``is_overridden``, without mutating the
        original ``y_hat`` — the statistical baseline stays visible for FVA.
        """
        active = self.active_overrides().select(["unique_id", "ds", pl.col("override_y_hat")])
        joined = forecasts.join(active, on=["unique_id", "ds"], how="left")
        return joined.with_columns(
            pl.coalesce(["override_y_hat", "y_hat"]).alias("y_hat_final"),
            pl.col("override_y_hat").is_not_null().alias("is_overridden"),
        ).drop("override_y_hat")
