"""The seven standard tables every adapter must produce, and their validator.

Units: timestamps are timezone-aware UTC; prices are USD per contract; strikes are in
underlying units. An adapter is finished when its output passes ``validate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Kind = Literal["str", "float", "int", "datetime", "bool"]


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    nullable: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        Table(
            "instruments",
            (
                Column("instrument_id", "str"),
                Column("venue", "str"),
                Column("underlying_id", "str"),
                Column("asset_class", "str"),
                Column("payoff_type", "str"),
                Column("strike", "float", nullable=True),
                Column("strike_upper", "float", nullable=True),
                Column("expiry_utc", "datetime"),
                Column("settlement_rule", "str"),
                Column("multiplier", "float"),
                Column("currency", "str"),
            ),
            key=("instrument_id",),
        ),
        Table(
            "quotes",
            (
                Column("ts_utc", "datetime"),
                Column("instrument_id", "str"),
                Column("bid", "float", nullable=True),
                Column("ask", "float", nullable=True),
                Column("bid_size", "float", nullable=True),
                Column("ask_size", "float", nullable=True),
                Column("source", "str"),
                Column("synchronised", "bool"),
            ),
            key=("ts_utc", "instrument_id", "source"),
        ),
        Table(
            "trades",
            (
                Column("ts_utc", "datetime"),
                Column("instrument_id", "str"),
                Column("price", "float"),
                Column("size", "float"),
                Column("aggressor_side", "str", nullable=True),
                Column("trade_id", "str"),
            ),
            key=("trade_id",),
        ),
        Table(
            "settlements",
            (
                Column("instrument_id", "str"),
                Column("settlement_value", "float"),
                Column("underlying_settlement_price", "float", nullable=True),
                Column("source", "str"),
            ),
            key=("instrument_id",),
        ),
        Table(
            "underlyings",
            (
                Column("ts_utc", "datetime"),
                Column("underlying_id", "str"),
                Column("spot", "float", nullable=True),
                Column("forward", "float", nullable=True),
                Column("index_value", "float", nullable=True),
                Column("source", "str"),
            ),
            key=("ts_utc", "underlying_id", "source"),
        ),
        Table(
            "rates",
            (
                Column("date", "datetime"),
                Column("tenor_days", "int"),
                Column("rate", "float"),
                Column("source", "str"),
            ),
            key=("date", "tenor_days", "source"),
        ),
        Table(
            "fees",
            (
                Column("venue", "str"),
                Column("product", "str"),
                Column("effective_from", "datetime"),
                Column("formula", "str"),
                Column("parameters", "str"),
            ),
            key=("venue", "product", "effective_from"),
        ),
    )
}


class SchemaError(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, table: str, problems: list[str]):
        self.table = table
        self.problems = problems
        super().__init__(f"{table}: " + "; ".join(problems))


def empty(table: str) -> pd.DataFrame:
    """An empty frame with the table's columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TABLES[table].column_names})


def validate(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Check ``df`` against the named table; return it with columns in standard order."""
    spec = TABLES[table]
    problems: list[str] = []

    missing = [c for c in spec.column_names if c not in df.columns]
    extra = [c for c in df.columns if c not in spec.column_names]
    if missing:
        problems.append(f"missing columns {missing}")
    if extra:
        problems.append(f"unexpected columns {extra}")
    if problems:
        raise SchemaError(table, problems)

    for col in spec.columns:
        series = df[col.name]
        nulls = series.isna()
        if not col.nullable and nulls.any():
            problems.append(f"{col.name}: {int(nulls.sum())} null values")
        values = series[~nulls]
        if values.empty:
            continue
        problems.extend(_check_kind(col, values))

    if not df.empty and df.duplicated(subset=list(spec.key)).any():
        problems.append(f"duplicate keys on {list(spec.key)}")

    if table == "quotes":
        problems.extend(_check_quotes(df))

    if problems:
        raise SchemaError(table, problems)
    return df[spec.column_names].reset_index(drop=True)


def _check_kind(col: Column, values: pd.Series) -> list[str]:
    if col.kind == "datetime":
        if not pd.api.types.is_datetime64_any_dtype(values):
            return [f"{col.name}: expected datetime, got {values.dtype}"]
        tz = getattr(values.dt, "tz", None)
        if tz is None or str(tz) != "UTC":
            return [f"{col.name}: timestamps must be timezone-aware UTC (got tz={tz})"]
        return []
    if col.kind == "float":
        if not pd.api.types.is_numeric_dtype(values):
            return [f"{col.name}: expected numeric, got {values.dtype}"]
        return []
    if col.kind == "int":
        if not pd.api.types.is_integer_dtype(values):
            return [f"{col.name}: expected integer, got {values.dtype}"]
        return []
    if col.kind == "bool":
        if not pd.api.types.is_bool_dtype(values):
            return [f"{col.name}: expected bool, got {values.dtype}"]
        return []
    if not all(isinstance(v, str) for v in values):
        return [f"{col.name}: expected strings"]
    return []


def _check_quotes(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    bid = pd.to_numeric(df["bid"], errors="coerce")
    ask = pd.to_numeric(df["ask"], errors="coerce")
    if (bid < 0).any() or (ask < 0).any():
        problems.append("negative bid or ask")
    crossed = (bid > ask) & bid.notna() & ask.notna()
    if crossed.any():
        problems.append(f"{int(crossed.sum())} crossed quotes (bid > ask)")
    return problems
