"""Raw SVI smile for one expiry (Gatheral 2004; Gatheral & Jacquier 2014).

Total implied variance as a function of log-moneyness k = ln(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Constraints: b >= 0, |rho| < 1, s > 0, and a + b * s * sqrt(1 - rho^2) >= 0 (w >= 0).
``butterfly_g`` evaluates Gatheral's density condition g(k) >= 0, which rules out
butterfly arbitrage (a negative implied density).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

F64 = NDArray[np.floating]


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    s: float
    t: float  # year fraction to expiry

    def w(self, k: ArrayLike) -> F64:
        x = np.asarray(k, float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.s * self.s))

    def dw(self, k: ArrayLike) -> F64:
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.s * self.s))

    def d2w(self, k: ArrayLike) -> F64:
        x = np.asarray(k, float) - self.m
        return self.b * self.s * self.s / (x * x + self.s * self.s) ** 1.5

    def vol(self, k: ArrayLike) -> F64:
        return np.sqrt(np.maximum(self.w(k), 0.0) / self.t)

    def vol_at_strike(self, strike: ArrayLike, forward: float) -> F64:
        return self.vol(np.log(np.asarray(strike, float) / forward))

    def dvol_dstrike(self, strike: ArrayLike, forward: float) -> F64:
        """d sigma / dK = w'(k) / (2 sigma T K)."""
        strike_ = np.asarray(strike, float)
        k = np.log(strike_ / forward)
        sigma = self.vol(k)
        return self.dw(k) / (2.0 * sigma * self.t * strike_)

    def butterfly_g(self, k: ArrayLike) -> F64:
        k_ = np.asarray(k, float)
        w, dw, d2w = self.w(k_), self.dw(k_), self.d2w(k_)
        return (1 - k_ * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2

    def is_butterfly_free(self, k_grid: ArrayLike | None = None) -> bool:
        grid = np.linspace(-1.5, 1.5, 601) if k_grid is None else np.asarray(k_grid, float)
        return bool(np.all(self.butterfly_g(grid) >= -1e-10))


def fit_svi(
    k: ArrayLike,
    iv: ArrayLike,
    t: float,
    weights: ArrayLike | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> SVI:
    """Least-squares fit of total variance with box bounds and a penalty on w_min < 0.

    ``k``: log-moneyness; ``iv``: implied vols; ``weights``: e.g. vega or 1/spread.
    Several random starts guard against local minima; the best fit is returned.
    """
    k_ = np.asarray(k, float)
    w_obs = np.asarray(iv, float) ** 2 * t
    ok = np.isfinite(k_) & np.isfinite(w_obs)
    k_, w_obs = k_[ok], w_obs[ok]
    wt = np.ones_like(k_) if weights is None else np.asarray(weights, float)[ok]
    wt = np.sqrt(wt / wt.mean())
    if k_.size < 5:
        raise ValueError("SVI needs at least five valid quotes")

    # Short expiries have total variances near 1e-4; fit in units of their mean so the
    # optimiser's tolerances mean something, then scale a and b back.
    scale = float(w_obs.mean())
    w_obs = w_obs / scale
    w_max = float(w_obs.max())
    span = float(k_.max() - k_.min())
    lo = np.array([-w_max, 1e-8, -0.999, k_.min() - span, 1e-5])
    hi = np.array([w_max, 100.0 * w_max / max(span, 1e-6), 0.999, k_.max() + span, span + 1.0])

    def resid(p: F64) -> F64:
        a, b, rho, m, s = p
        x = k_ - m
        model = a + b * (rho * x + np.sqrt(x * x + s * s))
        w_min = a + b * s * np.sqrt(1 - rho * rho)
        penalty = 1e3 * max(0.0, -w_min)
        return np.append(wt * (model - w_obs), penalty)

    rng = np.random.default_rng(seed)
    exact_cost = 1e-14 * k_.size
    best: tuple[float, F64] | None = None
    for i in range(n_starts):
        x0 = (
            np.array([0.5 * w_obs.min(), w_max / max(span, 1e-6), -0.3, 0.0, 0.1 * span])
            if i == 0
            else lo + rng.random(5) * (hi - lo)
        )
        res = least_squares(
            resid, np.clip(x0, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
        cost = float(res.cost)
        if best is None or cost < best[0]:
            best = (cost, res.x)
        if best[0] < exact_cost:  # an essentially exact fit; more starts cannot improve it
            break
    assert best is not None
    a, b, rho, m, s = best[1]
    return SVI(float(a * scale), float(b * scale), float(rho), float(m), float(s), float(t))
