"""Instruments: a payoff plus everything needed to match it across venues."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from derivkit.core.payoffs import Digital, Payoff, Range, Spread, Vanilla, payoff_type


class AssetClass(StrEnum):
    INDEX = "index"
    EQUITY = "equity"
    CRYPTO = "crypto"


class SettlementKind(StrEnum):
    CLOSE = "close"  # official closing value (e.g. S&P 500 4pm close)
    OPEN_QUOTATION = "open_quotation"  # AM-settled index options (special opening quotation)
    TWAP = "twap"  # time-weighted average over ``window_seconds`` before expiry
    TRIMMED_MEAN = "trimmed_mean"  # mean of observations after trimming each tail


@dataclass(frozen=True)
class SettlementRule:
    """How the terminal price is measured. Two instruments hedge each other cleanly only if
    their rules match; otherwise the difference is settlement basis and must be measured."""

    kind: SettlementKind
    source: str  # e.g. "kalshi:sp500_close", "cfbenchmarks:brti", "deribit:btc_usd_index"
    window_seconds: int = 0
    trim_fraction: float = 0.0

    def matches(self, other: SettlementRule) -> bool:
        return self == other

    def describe(self) -> str:
        if self.kind is SettlementKind.TWAP:
            return f"twap:{self.window_seconds}s:{self.source}"
        if self.kind is SettlementKind.TRIMMED_MEAN:
            return f"trimmed_mean:{self.window_seconds}s:{self.trim_fraction}:{self.source}"
        return f"{self.kind.value}:{self.source}"


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    venue: str
    underlying_id: str
    asset_class: AssetClass
    payoff: Payoff
    expiry_utc: datetime
    settlement: SettlementRule
    multiplier: float = 1.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.expiry_utc.tzinfo is None:
            raise ValueError(f"{self.instrument_id}: expiry_utc must be timezone-aware")

    def to_record(self) -> dict[str, Any]:
        """Row for the standard ``instruments`` table."""
        strike, strike_upper = _strikes(self.payoff)
        return {
            "instrument_id": self.instrument_id,
            "venue": self.venue,
            "underlying_id": self.underlying_id,
            "asset_class": self.asset_class.value,
            "payoff_type": payoff_type(self.payoff),
            "strike": strike,
            "strike_upper": strike_upper,
            "expiry_utc": self.expiry_utc,
            "settlement_rule": self.settlement.describe(),
            "multiplier": float(self.multiplier),
            "currency": self.currency,
        }


def _strikes(payoff: Payoff) -> tuple[float | None, float | None]:
    match payoff:
        case Vanilla(strike=k) | Digital(strike=k):
            return float(k), None
        case Range(lower=lo, upper=hi):
            return lo, hi
        case Spread():
            return None, None
    raise TypeError(f"Unknown payoff {payoff!r}")
