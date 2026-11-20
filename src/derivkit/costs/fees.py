"""Venue fee models, versioned by effective date.

Kalshi (general schedule): fee = round_up_to_cent(rate * multiplier * C * P * (1 - P)),
taker rate 0.07, maker rate 0.0175. Some series use a different multiplier (index markets
are reported to use half); set it per product from the published schedule, never by guess.
Source: https://kalshi.com/docs/kalshi-fee-schedule.pdf
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TypeVar


def ceil_cent(x: float) -> float:
    """Round up to the next cent, robust to float noise (0.0175 * 100 -> 1.75, not 1.76)."""
    return math.ceil(round(x * 100.0, 9)) / 100.0


@dataclass(frozen=True)
class KalshiFee:
    effective_from: date
    taker_rate: float = 0.07
    maker_rate: float = 0.0175
    multiplier: float = 1.0

    def fee(self, contracts: float, price: float, maker: bool = False) -> float:
        """Total fee in dollars for one order of ``contracts`` at ``price`` (dollars)."""
        if not 0.0 <= price <= 1.0:
            raise ValueError(f"Kalshi price must be in [0, 1], got {price}")
        rate = self.maker_rate if maker else self.taker_rate
        return ceil_cent(rate * self.multiplier * contracts * price * (1.0 - price))


@dataclass(frozen=True)
class DeribitOptionFee:
    """Per-contract fee as a fraction of the underlying, capped at a fraction of premium.

    Default parameters are placeholders to be checked against Deribit's current schedule.
    """

    effective_from: date
    underlying_rate: float = 0.0003
    premium_cap: float = 0.125

    def fee(self, contracts: float, premium_usd: float, underlying_usd: float) -> float:
        per = min(self.underlying_rate * underlying_usd, self.premium_cap * premium_usd)
        return contracts * per


@dataclass(frozen=True)
class PerContractFee:
    effective_from: date
    amount: float  # dollars per contract

    def fee(self, contracts: float, *_: float) -> float:
        return contracts * self.amount


Fee = KalshiFee | DeribitOptionFee | PerContractFee
F = TypeVar("F", KalshiFee, DeribitOptionFee, PerContractFee)


def fee_in_force(schedule: Sequence[F], on: date) -> F:
    """The latest schedule entry whose ``effective_from`` is on or before ``on``."""
    eligible = [f for f in schedule if f.effective_from <= on]
    if not eligible:
        raise LookupError(f"No fee schedule in force on {on}")
    return max(eligible, key=lambda f: f.effective_from)
