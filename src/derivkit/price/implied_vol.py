"""Black-76 implied volatility by bracketed root finding."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from derivkit.price.black import black_price

VOL_LO, VOL_HI = 1e-6, 10.0


def implied_vol(
    price: ArrayLike,
    f: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    df: ArrayLike = 1.0,
    call: bool | ArrayLike = True,
) -> NDArray[np.float64]:
    """Implied vol for each option; NaN where the price breaks no-arbitrage bounds."""
    p, f_, k_, t_, df_, c_ = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(f, float),
        np.asarray(k, float),
        np.asarray(t, float),
        np.asarray(df, float),
        np.asarray(call, bool),
    )
    out = np.full(p.shape, np.nan)
    for idx in np.ndindex(p.shape):
        out[idx] = _one(
            float(p[idx]),
            float(f_[idx]),
            float(k_[idx]),
            float(t_[idx]),
            float(df_[idx]),
            bool(c_[idx]),
        )
    return out


def _one(price: float, f: float, k: float, t: float, df: float, call: bool) -> float:
    if not np.isfinite(price) or t <= 0 or price <= 0:
        return np.nan
    lower = df * max(f - k, 0.0) if call else df * max(k - f, 0.0)
    upper = df * f if call else df * k
    if price <= lower or price >= upper:
        return np.nan

    def err(sigma: float) -> float:
        return float(black_price(f, k, t, sigma, df, call)) - price

    lo, hi = err(VOL_LO), err(VOL_HI)
    if lo > 0 or hi < 0:
        return np.nan
    return cast(float, brentq(err, VOL_LO, VOL_HI, xtol=1e-12, maxiter=200))
