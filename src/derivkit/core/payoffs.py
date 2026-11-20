"""Payoffs as functions of the terminal underlying price.

Every payoff here depends only on S_T, so any of them can be priced off a volatility
surface and bounded by a static portfolio of listed calls and puts (see ``replicate``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Vanilla:
    """European call or put: max(S - K, 0) or max(K - S, 0)."""

    option_type: OptionType
    strike: float


@dataclass(frozen=True)
class Digital:
    """Cash-or-nothing digital paying 1.

    ``above=True`` pays 1 if S_T > strike; ``above=False`` pays 1 if S_T < strike.
    Kalshi's "7,400 or above" contract is ``Digital(7399.9999, above=True)``.
    """

    strike: float
    above: bool = True


@dataclass(frozen=True)
class Range:
    """Pays 1 if lower <= S_T <= upper. ``None`` leaves that side open.

    Kalshi's "7,625 to 7,649.9999" bucket is ``Range(7625, 7649.9999)``.
    """

    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("Range needs at least one finite bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"Range lower {self.lower} exceeds upper {self.upper}")


@dataclass(frozen=True)
class Spread:
    """Weighted sum of other payoffs, e.g. a vertical call spread."""

    legs: tuple[tuple[float, Payoff], ...]


Payoff = Vanilla | Digital | Range | Spread


def payoff_value(payoff: Payoff, s: ArrayLike) -> NDArray[np.float64]:
    """Payoff at terminal prices ``s`` (vectorised)."""
    s_arr = np.asarray(s, dtype=float)
    match payoff:
        case Vanilla(option_type=OptionType.CALL, strike=k):
            return np.maximum(s_arr - k, 0.0)
        case Vanilla(option_type=OptionType.PUT, strike=k):
            return np.maximum(k - s_arr, 0.0)
        case Digital(strike=k, above=True):
            return (s_arr > k).astype(float)
        case Digital(strike=k, above=False):
            return (s_arr < k).astype(float)
        case Range(lower=lo, upper=hi):
            inside = np.ones_like(s_arr, dtype=bool)
            if lo is not None:
                inside &= s_arr >= lo
            if hi is not None:
                inside &= s_arr <= hi
            return inside.astype(float)
        case Spread(legs=legs):
            total = np.zeros_like(s_arr, dtype=float)
            for weight, leg in legs:
                total += weight * payoff_value(leg, s_arr)
            return total
    raise TypeError(f"Unknown payoff {payoff!r}")


def payoff_type(payoff: Payoff) -> str:
    """Short name stored in the ``instruments`` table."""
    match payoff:
        case Vanilla(option_type=t):
            return t.value
        case Digital(above=True):
            return "digital_above"
        case Digital(above=False):
            return "digital_below"
        case Range():
            return "range"
        case Spread():
            return "spread"
    raise TypeError(f"Unknown payoff {payoff!r}")
