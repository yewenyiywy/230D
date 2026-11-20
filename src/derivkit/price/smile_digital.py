"""Digital prices consistent with the volatility smile.

A digital paying 1 if S_T > K is the negative strike-derivative of the call price.
On a smile sigma(K), differentiating C(K, sigma(K)) gives

    D(K) = df * N(d2) - vega(K) * dsigma/dK

The second term is what the flat-vol Black-Scholes digital misses. With a typical equity
skew (dsigma/dK < 0) it makes upside digitals more expensive than N(d2) suggests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calibrate.svi import SVI
from derivkit.price.black import black_digital, black_price, black_vega

F64 = NDArray[np.floating]


def smile_digital(
    strike: ArrayLike,
    forward: float,
    t: float,
    sigma: Callable[[F64], F64],
    dsigma_dk: Callable[[F64], F64],
    df: float = 1.0,
    above: bool = True,
) -> F64:
    """Smile-consistent digital. ``sigma`` and ``dsigma_dk`` take strikes (not log-moneyness)."""
    k = np.asarray(strike, float)
    vol = sigma(k)
    up = black_digital(forward, k, t, vol, df, above=True) - black_vega(
        forward, k, t, vol, df
    ) * dsigma_dk(k)
    return up if above else df - up


def svi_digital(
    strike: ArrayLike, forward: float, svi: SVI, df: float = 1.0, above: bool = True
) -> F64:
    """Smile-consistent digital using an SVI fit for the same expiry."""
    return smile_digital(
        strike,
        forward,
        svi.t,
        lambda k: svi.vol_at_strike(k, forward),
        lambda k: svi.dvol_dstrike(k, forward),
        df,
        above,
    )


def svi_call(strike: ArrayLike, forward: float, svi: SVI, df: float = 1.0) -> F64:
    k = np.asarray(strike, float)
    return black_price(forward, k, svi.t, svi.vol_at_strike(k, forward), df, call=True)


def finite_difference_digital(strike: ArrayLike, call_price: Callable[[F64], F64], h: float) -> F64:
    """Model-free check: -(C(K + h) - C(K - h)) / 2h."""
    k = np.asarray(strike, float)
    return -(call_price(k + h) - call_price(k - h)) / (2.0 * h)
