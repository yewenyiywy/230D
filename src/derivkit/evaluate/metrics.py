"""Out-of-sample evaluation: net P&L, risk, attribution and confidence intervals.

Trades settling on the same date are not independent (all buckets of one S&P close move
together), so intervals come from a block bootstrap that resamples whole settlement dates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from derivkit.backtest.engine import ATTRIBUTION

TRADING_DAYS = 252


@dataclass(frozen=True)
class Summary:
    n_trades: int
    n_tier1: int
    n_tier2: int
    n_days: int
    total_pnl: float
    mean_pnl_per_trade: float
    ci_low: float
    ci_high: float
    hit_rate: float
    sharpe_annualised: float
    max_drawdown: float
    attribution: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def settle_dates(trades: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(trades["expiry_utc"], utc=True).dt.tz_convert("America/New_York").dt.date


def daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby(settle_dates(trades))["pnl"].sum().sort_index()


def max_drawdown(daily: pd.Series) -> float:
    if daily.empty:
        return 0.0
    equity = daily.cumsum()
    return float((equity.cummax() - equity).max())


def block_bootstrap_mean(
    trades: pd.DataFrame, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for mean P&L per trade, resampling settlement dates."""
    if trades.empty:
        return (np.nan, np.nan)
    days = settle_dates(trades)
    by_day = trades.assign(_day=days).groupby("_day")["pnl"].agg(["sum", "count"])
    sums, counts = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by_day), size=(n_boot, len(by_day)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarise(trades: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> Summary:
    """Summary over settled trades only."""
    t = trades[trades["settled"]] if not trades.empty else trades
    daily = daily_pnl(t)
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    lo, hi = block_bootstrap_mean(t, n_boot=n_boot, seed=seed)
    return Summary(
        n_trades=len(t),
        n_tier1=int((t["tier"] == 1).sum()) if not t.empty else 0,
        n_tier2=int((t["tier"] == 2).sum()) if not t.empty else 0,
        n_days=len(daily),
        total_pnl=float(t["pnl"].sum()) if not t.empty else 0.0,
        mean_pnl_per_trade=float(t["pnl"].mean()) if not t.empty else float("nan"),
        ci_low=lo,
        ci_high=hi,
        hit_rate=float((t["pnl"] > 0).mean()) if not t.empty else float("nan"),
        sharpe_annualised=sharpe,
        max_drawdown=max_drawdown(daily),
        attribution={k: float(t[k].sum()) if not t.empty else 0.0 for k in ATTRIBUTION},
    )
