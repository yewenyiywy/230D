"""Adapter contract.

An adapter has two halves:

* ``fetch_*`` methods talk to a venue's API and return raw JSON or files unchanged
  (stored as-is so a parsing fix never needs a re-download);
* ``normalize`` turns raw payloads into the standard tables, with no network access.

Only ``normalize`` is exercised in CI, against recorded fixtures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from derivkit.core.instruments import Instrument
from derivkit.core.schemas import TABLES, empty, validate


@dataclass
class Tables:
    """The standard tables produced by one normalize call. Missing tables are empty."""

    instruments: pd.DataFrame = field(default_factory=lambda: empty("instruments"))
    quotes: pd.DataFrame = field(default_factory=lambda: empty("quotes"))
    trades: pd.DataFrame = field(default_factory=lambda: empty("trades"))
    settlements: pd.DataFrame = field(default_factory=lambda: empty("settlements"))
    underlyings: pd.DataFrame = field(default_factory=lambda: empty("underlyings"))
    rates: pd.DataFrame = field(default_factory=lambda: empty("rates"))
    fees: pd.DataFrame = field(default_factory=lambda: empty("fees"))

    def validated(self) -> Tables:
        """Validate every non-empty table; raises ``SchemaError`` on the first bad table."""
        out = Tables()
        for name in TABLES:
            df: pd.DataFrame = getattr(self, name)
            setattr(out, name, validate(df, name) if not df.empty else df)
        return out


class Adapter(ABC):
    """Base class for venue adapters."""

    venue: str

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> Tables:
        """Turn a dict of raw payloads (keyed by payload type) into standard tables."""

    def instruments_frame(self, instruments: list[Instrument]) -> pd.DataFrame:
        if not instruments:
            return empty("instruments")
        df = pd.DataFrame([i.to_record() for i in instruments])
        df["expiry_utc"] = pd.to_datetime(df["expiry_utc"], utc=True)
        return df


def utc(series: pd.Series, unit: Literal["s", "ms", "us", "ns"] | None = None) -> pd.Series:
    """Parse a column of timestamps (ISO strings or epoch numbers) to UTC."""
    if unit is not None:
        return pd.to_datetime(series, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True, format="ISO8601")
