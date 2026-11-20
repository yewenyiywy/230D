"""Model-free bounds for digitals and ranges from executable call-spread quotes.

A call spread long K1, short K2 (K1 < K2), scaled by 1 / (K2 - K1), pays a ramp from 0
at K1 to 1 at K2. For a digital paying 1 if S_T > K:

* the ramp dominates the digital if K2 <= K  -> buying it (at ask/bid) gives an upper bound;
* the ramp is dominated if K1 >= K            -> selling it gives a lower bound.

``tick`` handles underlyings quoted on a price grid. The S&P 500 prints to the cent, so
"above 7399.9999" means S_T >= 7400.00, and a spread ending exactly at 7400 still
dominates. With ``tick=0`` the underlying is treated as continuous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from derivkit.core.payoffs import Digital, Payoff, Range
from derivkit.replicate.chain import OptionChain

_EPS = 1e-9


@dataclass(frozen=True)
class Bound:
    value: float
    k1: float | None  # strikes of the spread achieving the bound (None = trivial bound)
    k2: float | None


@dataclass(frozen=True)
class Bounds:
    lower: Bound
    upper: Bound

    @property
    def width(self) -> float:
        return self.upper.value - self.lower.value

    def contains(self, x: float, tol: float = 1e-12) -> bool:
        return self.lower.value - tol <= x <= self.upper.value + tol


def _thresholds(strike: float, tick: float, strict: bool) -> tuple[float, float]:
    """(last price paying 0, first price paying 1) for an 'above' digital."""
    if tick <= 0:
        return strike, strike
    if strict:  # pays if S > strike
        first_one = math.floor(strike / tick + _EPS) * tick + tick
    else:  # pays if S >= strike
        first_one = math.ceil(strike / tick - _EPS) * tick
    return first_one - tick, first_one


def digital_above_bounds(
    chain: OptionChain, strike: float, tick: float = 0.0, strict: bool = True
) -> Bounds:
    """Bounds on a digital paying 1 if S_T > strike (``strict``) or S_T >= strike."""
    last_zero, first_one = _thresholds(strike, tick, strict)
    ks, bid, ask = chain.strikes, chain.call_bid, chain.call_ask

    upper = Bound(chain.df, None, None)  # a zero-coupon bond always dominates
    lower = Bound(0.0, None, None)
    n = len(ks)
    for i in range(n):
        for j in range(i + 1, n):
            k1, k2 = ks[i], ks[j]
            width = k2 - k1
            if k2 <= first_one + _EPS and np.isfinite(ask[i]) and np.isfinite(bid[j]):
                cost = (ask[i] - bid[j]) / width
                if cost < upper.value:
                    upper = Bound(float(cost), float(k1), float(k2))
            if k1 >= last_zero - _EPS and np.isfinite(bid[i]) and np.isfinite(ask[j]):
                proceeds = (bid[i] - ask[j]) / width
                if proceeds > lower.value:
                    lower = Bound(float(proceeds), float(k1), float(k2))
    return Bounds(lower, upper)


def payoff_bounds(chain: OptionChain, payoff: Payoff, tick: float = 0.0) -> Bounds:
    """Bounds for digitals (either side) and ranges, built from the 'above' case."""
    df = chain.df
    match payoff:
        case Digital(strike=k, above=True):
            return digital_above_bounds(chain, k, tick, strict=True)
        case Digital(strike=k, above=False):
            # 1{S < K} = 1 - 1{S >= K}
            b = digital_above_bounds(chain, k, tick, strict=False)
            return Bounds(
                Bound(df - b.upper.value, b.upper.k1, b.upper.k2),
                Bound(df - b.lower.value, b.lower.k1, b.lower.k2),
            )
        case Range(lower=lo, upper=hi):
            # 1{lo <= S <= hi} = 1{S >= lo} - 1{S > hi}
            if lo is None:
                assert hi is not None  # Range rejects two open sides
                # 1{S <= hi} = 1 - 1{S > hi}
                above_hi = digital_above_bounds(chain, hi, tick, strict=True)
                return Bounds(
                    Bound(df - above_hi.upper.value, None, None),
                    Bound(df - above_hi.lower.value, None, None),
                )
            at_or_above_lo = digital_above_bounds(chain, lo, tick, strict=False)
            if hi is None:
                return at_or_above_lo
            above_hi = digital_above_bounds(chain, hi, tick, strict=True)
            lower = max(0.0, at_or_above_lo.lower.value - above_hi.upper.value)
            upper = min(df, at_or_above_lo.upper.value - above_hi.lower.value)
            return Bounds(Bound(lower, None, None), Bound(upper, None, None))
    raise TypeError(f"Pairwise bounds not implemented for {payoff!r}; use replicate.lp")
