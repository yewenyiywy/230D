"""Market state for one expiry at one decision time: forward, discount factor, smile.

The forward and discount factor are backed out of the option chain itself by put-call
parity, C - P = df * (F - K): a straight-line fit of (C - P) against K near the money
gives slope -df and intercept df * F. That keeps the pricing self-consistent with the
quotes being hedged, without needing a separate rates or dividend feed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from derivkit.calibrate.svi import SVI, fit_svi
from derivkit.core.payoffs import Digital, Payoff, Range
from derivkit.price.implied_vol import implied_vol
from derivkit.price.smile_digital import svi_digital
from derivkit.replicate.chain import OptionChain

SECONDS_PER_YEAR = 365.0 * 24 * 3600


class MarketStateError(ValueError):
    """The chain does not support a reliable forward or smile; skip this decision."""


@dataclass(frozen=True)
class MarketState:
    chain: OptionChain
    forward: float
    df: float
    t: float  # year fraction, ACT/365
    svi: SVI

    def fair_value(self, payoff: Payoff) -> float:
        """Smile-consistent model price of a digital or range paying 1."""
        f, s, df = self.forward, self.svi, self.df
        match payoff:
            case Digital(strike=k, above=above):
                return float(svi_digital(k, f, s, df, above=above))
            case Range(lower=lo, upper=hi):
                up_lo = df if lo is None else float(svi_digital(lo, f, s, df))
                up_hi = 0.0 if hi is None else float(svi_digital(hi, f, s, df))
                return up_lo - up_hi
        raise TypeError(f"No fair-value rule for {payoff!r}")


def mids(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(bid) & np.isfinite(ask), 0.5 * (bid + ask), np.nan)


def parity_forward(chain: OptionChain, n_strikes: int = 10) -> tuple[float, float]:
    """(forward, discount factor) from the ``n_strikes`` strikes nearest the money."""
    if chain.put_bid is None or chain.put_ask is None:
        raise MarketStateError("parity needs puts")
    c = mids(chain.call_bid, chain.call_ask)
    p = mids(chain.put_bid, chain.put_ask)
    ok = np.isfinite(c) & np.isfinite(p)
    if ok.sum() < 3:
        raise MarketStateError("fewer than three strikes with both call and put quotes")
    k, diff = chain.strikes[ok], (c - p)[ok]
    atm = k[np.argmin(np.abs(diff))]
    near = np.argsort(np.abs(k - atm))[:n_strikes]
    slope, intercept = np.polyfit(k[near], diff[near], 1)
    df = -float(slope)
    if not 0.5 < df <= 1.01:
        raise MarketStateError(f"implausible discount factor {df:.4f} from parity")
    return float(intercept) / df, min(df, 1.0)


def build_state(chain: OptionChain, t: float, min_quotes: int = 5) -> MarketState:
    """Forward and df from parity, then SVI on out-of-the-money mid implied vols."""
    if t <= 0:
        raise MarketStateError("expiry is not after the decision time")
    forward, df = parity_forward(chain)
    chain = replace(chain, df=df)
    k = chain.strikes
    call_mid = mids(chain.call_bid, chain.call_ask)
    use_call = k >= forward
    price, bid = call_mid.copy(), chain.call_bid
    if chain.put_bid is not None and chain.put_ask is not None:
        put_mid = mids(chain.put_bid, chain.put_ask)
        price = np.where(use_call, call_mid, put_mid)
        bid = np.where(use_call, chain.call_bid, chain.put_bid)
    else:
        use_call = np.ones_like(k, dtype=bool)
    # A zero bid means the mid is just half the ask, not a price: drop those quotes.
    price = np.where(bid > 0, price, np.nan)
    iv = implied_vol(price, forward, k, t, df, call=use_call)
    ok = np.isfinite(iv)
    if ok.sum() < min_quotes:
        raise MarketStateError(f"only {int(ok.sum())} usable implied vols")
    svi = fit_svi(np.log(k[ok] / forward), iv[ok], t)
    return MarketState(chain=chain, forward=forward, df=df, t=t, svi=svi)
