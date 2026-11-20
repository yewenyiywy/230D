"""Static hedges: a bond plus call positions held to expiry.

For a digital paying 1 above a threshold, the hedge is a call spread scaled to pay 1:
either the spread that sets a replication bound (tier-1 trades) or the listed strikes
either side of the threshold (tier-2 trades). A range is two such spreads; a 'below'
digital is a bond minus the 'above' spread.

Quantities are per one target contract. Call quantities are in units of the underlying;
divide by the venue multiplier to count listed option contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivkit.core.payoffs import Digital, Payoff, Range
from derivkit.replicate.bounds import Bounds, tick_thresholds
from derivkit.replicate.chain import OptionChain


@dataclass(frozen=True)
class Leg:
    kind: str  # "call" or "bond"
    strike: float  # 0 for the bond
    qty: float  # positive = long


@dataclass(frozen=True)
class Hedge:
    legs: tuple[Leg, ...]

    def scaled(self, factor: float) -> Hedge:
        return Hedge(tuple(Leg(leg.kind, leg.strike, leg.qty * factor) for leg in self.legs))

    def payoff(self, s: float) -> float:
        total = 0.0
        for leg in self.legs:
            total += leg.qty * (1.0 if leg.kind == "bond" else max(s - leg.strike, 0.0))
        return total

    def prices(self, chain: OptionChain) -> tuple[float, float]:
        """(execution value, mid value): buying pays the ask, selling receives the bid."""
        exec_value = mid_value = 0.0
        for leg in self.legs:
            if leg.kind == "bond":
                exec_value += leg.qty * chain.df
                mid_value += leg.qty * chain.df
                continue
            i = int(np.searchsorted(chain.strikes, leg.strike))
            if i >= len(chain.strikes) or chain.strikes[i] != leg.strike:
                raise KeyError(f"strike {leg.strike} not listed")
            bid, ask = float(chain.call_bid[i]), float(chain.call_ask[i])
            if not (np.isfinite(bid) and np.isfinite(ask)):
                raise KeyError(f"strike {leg.strike} has no two-sided quote")
            exec_value += leg.qty * (ask if leg.qty > 0 else bid)
            mid_value += leg.qty * 0.5 * (bid + ask)
        return exec_value, mid_value

    def option_units(self) -> float:
        """Total absolute call quantity (units of underlying), for commission charging."""
        return sum(abs(leg.qty) for leg in self.legs if leg.kind == "call")

    def describe(self) -> str:
        parts = []
        for leg in self.legs:
            name = "bond" if leg.kind == "bond" else f"C{leg.strike:g}"
            parts.append(f"{leg.qty:+.4g} {name}")
        return ", ".join(parts)


def _spread(k1: float, k2: float) -> list[Leg]:
    w = k2 - k1
    return [Leg("call", k1, 1.0 / w), Leg("call", k2, -1.0 / w)]


def _quoted(chain: OptionChain) -> np.ndarray:
    ok = np.isfinite(chain.call_bid) & np.isfinite(chain.call_ask)
    return chain.strikes[ok]


def straddling_spread(chain: OptionChain, threshold: float) -> tuple[float, float]:
    """Nearest quoted strikes K1 <= threshold < K2."""
    ks = _quoted(chain)
    below, above = ks[ks <= threshold], ks[ks > threshold]
    if below.size == 0 or above.size == 0:
        raise KeyError(f"no quoted strikes either side of {threshold}")
    return float(below.max()), float(above.min())


def replicating_hedge(chain: OptionChain, payoff: Payoff, tick: float = 0.0) -> Hedge:
    """Long portfolio that approximately replicates ``payoff`` with the nearest strikes."""
    match payoff:
        case Digital(strike=k, above=True):
            last_zero, _ = tick_thresholds(k, tick, strict=True)
            return Hedge(tuple(_spread(*straddling_spread(chain, last_zero))))
        case Digital(strike=k, above=False):
            last_zero, _ = tick_thresholds(k, tick, strict=False)
            legs = [Leg("bond", 0.0, 1.0)]
            legs += [
                Leg(x.kind, x.strike, -x.qty) for x in _spread(*straddling_spread(chain, last_zero))
            ]
            return Hedge(tuple(legs))
        case Range(lower=lo, upper=hi):
            legs: list[Leg] = []
            if lo is None:
                legs.append(Leg("bond", 0.0, 1.0))
            else:
                lo_zero, _ = tick_thresholds(lo, tick, strict=False)
                legs += _spread(*straddling_spread(chain, lo_zero))
            if hi is not None:
                hi_zero, _ = tick_thresholds(hi, tick, strict=True)
                legs += [
                    Leg(x.kind, x.strike, -x.qty)
                    for x in _spread(*straddling_spread(chain, hi_zero))
                ]
            return Hedge(tuple(legs))
    raise TypeError(f"No static hedge rule for {payoff!r}")


def bound_hedge(bounds: Bounds, side: str) -> Hedge | None:
    """The spread that set the lower (``side="lower"``) or upper bound of an 'above' digital.

    Returns None when the bound was trivial or combined (ranges), so the caller falls back
    to ``replicating_hedge``.
    """
    b = bounds.lower if side == "lower" else bounds.upper
    if b.k1 is None or b.k2 is None:
        return None
    return Hedge(tuple(_spread(b.k1, b.k2)))
