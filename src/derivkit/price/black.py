"""Black-76 pricing on the forward (Black-Scholes is the special case F = S e^{(r-q)T}).

All functions are vectorised over numpy arrays. ``df`` is the discount factor to expiry.
Edge cases T <= 0 or sigma <= 0 return the discounted intrinsic value on the forward.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

F64 = NDArray[np.floating]


def _d1_d2(f: F64, k: F64, t: F64, sigma: F64) -> tuple[F64, F64]:
    vol_t = sigma * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(f / k) + 0.5 * vol_t**2) / vol_t
    return d1, d1 - vol_t


def _arrays(*xs: ArrayLike) -> list[F64]:
    return [np.asarray(x, dtype=float) for x in np.broadcast_arrays(*xs)]


def _degenerate(t: F64, sigma: F64) -> NDArray[np.bool_]:
    return (t <= 0) | (sigma <= 0)


def black_price(
    f: ArrayLike, k: ArrayLike, t: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0, call=True
) -> F64:
    """Black-76 price of a European call (``call=True``) or put."""
    f_, k_, t_, s_, df_ = _arrays(f, k, t, sigma, df)
    d1, d2 = _d1_d2(f_, k_, np.maximum(t_, 1e-300), np.maximum(s_, 1e-300))
    if call:
        px = df_ * (f_ * norm.cdf(d1) - k_ * norm.cdf(d2))
        intrinsic = df_ * np.maximum(f_ - k_, 0.0)
    else:
        px = df_ * (k_ * norm.cdf(-d2) - f_ * norm.cdf(-d1))
        intrinsic = df_ * np.maximum(k_ - f_, 0.0)
    return np.where(_degenerate(t_, s_), intrinsic, px)


def black_digital(
    f: ArrayLike, k: ArrayLike, t: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0, above=True
) -> F64:
    """Flat-volatility cash-or-nothing digital: df * N(d2) (above) or df * N(-d2) (below).

    This ignores the smile; see ``smile_digital`` for the price consistent with a skew.
    """
    f_, k_, t_, s_, df_ = _arrays(f, k, t, sigma, df)
    _, d2 = _d1_d2(f_, k_, np.maximum(t_, 1e-300), np.maximum(s_, 1e-300))
    px = df_ * norm.cdf(d2 if above else -d2)
    intrinsic = df_ * ((f_ > k_) if above else (f_ < k_)).astype(float)
    return np.where(_degenerate(t_, s_), intrinsic, px)


def black_vega(
    f: ArrayLike, k: ArrayLike, t: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> F64:
    """dPrice/dSigma, identical for calls and puts."""
    f_, k_, t_, s_, df_ = _arrays(f, k, t, sigma, df)
    d1, _ = _d1_d2(f_, k_, np.maximum(t_, 1e-300), np.maximum(s_, 1e-300))
    return np.where(_degenerate(t_, s_), 0.0, df_ * f_ * norm.pdf(d1) * np.sqrt(t_))


def black_delta(
    f: ArrayLike, k: ArrayLike, t: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0, call=True
) -> F64:
    """dPrice/dF."""
    f_, k_, t_, s_, df_ = _arrays(f, k, t, sigma, df)
    d1, _ = _d1_d2(f_, k_, np.maximum(t_, 1e-300), np.maximum(s_, 1e-300))
    delta = df_ * (norm.cdf(d1) if call else norm.cdf(d1) - 1.0)
    itm = (f_ > k_) if call else (f_ < k_)
    sign = 1.0 if call else -1.0
    return np.where(_degenerate(t_, s_), df_ * sign * itm, delta)


def black_gamma(
    f: ArrayLike, k: ArrayLike, t: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> F64:
    """d2Price/dF2, identical for calls and puts."""
    f_, k_, t_, s_, df_ = _arrays(f, k, t, sigma, df)
    d1, _ = _d1_d2(f_, k_, np.maximum(t_, 1e-300), np.maximum(s_, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = df_ * norm.pdf(d1) / (f_ * s_ * np.sqrt(t_))
    return np.where(_degenerate(t_, s_), 0.0, g)


def forward_from_spot(s: ArrayLike, t: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> F64:
    s_, t_, r_, q_ = _arrays(s, t, r, q)
    return s_ * np.exp((r_ - q_) * t_)
