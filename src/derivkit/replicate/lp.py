"""Super- and sub-replication of any terminal payoff by linear programming.

Portfolio: a zero-coupon bond (any sign) plus long and short positions in each listed
call and put, bought at the ask and sold at the bid. Its payoff is piecewise linear with
kinks at the strikes, so dominance only has to be checked at the strikes, at the payoff's
own breakpoints (both sides of a jump), at zero, and in the slope beyond the last strike.

This generalises the call-spread bounds in ``bounds.py`` to ranges, spreads, vanillas
quoted on another venue, or any payoff of S_T (the Carr-Madan idea with finitely many
strikes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from derivkit.core.payoffs import Digital, Payoff, Range, Spread, Vanilla, payoff_value
from derivkit.replicate.chain import OptionChain

F64 = NDArray[np.floating]


@dataclass(frozen=True)
class Replication:
    value: float  # cost to buy (super) or proceeds from selling (sub)
    bond: float
    calls: F64  # net quantity per strike (positive = long)
    puts: F64
    strikes: F64

    def payoff(self, s: F64) -> F64:
        s = np.asarray(s, float)[:, None]
        out = self.bond + np.maximum(s - self.strikes, 0.0) @ self.calls
        out = out + np.maximum(self.strikes - s, 0.0) @ self.puts
        return out


def _breakpoints(payoff: Payoff) -> list[float]:
    match payoff:
        case Vanilla(strike=k) | Digital(strike=k):
            return [k]
        case Range(lower=lo, upper=hi):
            return [x for x in (lo, hi) if x is not None]
        case Spread(legs=legs):
            return [b for _, leg in legs for b in _breakpoints(leg)]
    return []


def _grid(chain: OptionChain, payoff: Payoff, extra: Sequence[float]) -> F64:
    ks = chain.strikes
    pts = {0.0, *ks.tolist(), *extra}
    for b in _breakpoints(payoff):
        h = max(1e-9 * abs(b), 1e-12)
        pts.update({b - h, b, b + h})
    top = 2.0 * max(ks.max(), *(_breakpoints(payoff) or [0.0]))
    pts.add(top)
    return np.array(sorted(p for p in pts if p >= 0.0))


def _tail_slope(payoff: Payoff, top: float) -> float:
    s = np.array([top, 2.0 * top])
    v = payoff_value(payoff, s)
    return float((v[1] - v[0]) / top)


def replicate(
    chain: OptionChain,
    payoff: Payoff,
    side: str = "super",
    extra_points: Sequence[float] = (),
    max_position: float = 1e4,
) -> Replication:
    """Cheapest dominating (``side="super"``) or dearest dominated (``"sub"``) portfolio."""
    if side not in ("super", "sub"):
        raise ValueError("side must be 'super' or 'sub'")
    ks = chain.strikes
    n = len(ks)
    use_puts = chain.put_bid is not None and chain.put_ask is not None
    grid = _grid(chain, payoff, extra_points)
    target = payoff_value(payoff, grid)

    # Variables: [bond, call_long(n), call_short(n), put_long(n), put_short(n)]
    call_pay = np.maximum(grid[:, None] - ks[None, :], 0.0)
    put_pay = np.maximum(ks[None, :] - grid[:, None], 0.0)
    pay = np.hstack([np.ones((len(grid), 1)), call_pay, -call_pay, put_pay, -put_pay])

    def quotes(x: F64 | None) -> F64:
        return np.full(n, np.nan) if x is None else np.asarray(x, float)

    c_ask, c_bid = quotes(chain.call_ask), quotes(chain.call_bid)
    p_ask, p_bid = quotes(chain.put_ask), quotes(chain.put_bid)
    # Buying the portfolio (super): long legs pay the ask, short legs receive the bid.
    # Selling it (sub): long legs are sold at the bid, short legs bought back at the ask.
    leg_prices = (c_ask, c_bid, p_ask, p_bid) if side == "super" else (c_bid, c_ask, p_bid, p_ask)
    long_c, short_c, long_p, short_p = leg_prices
    value = np.concatenate([[chain.df], long_c, -short_c, long_p, -short_p])

    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for block, enabled in (
        (long_c, True),
        (short_c, True),
        (long_p, use_puts),
        (short_p, use_puts),
    ):
        for v in block:
            tradable = enabled and np.isfinite(v)
            bounds.append((0.0, max_position if tradable else 0.0))
    cost = np.where(np.isfinite(value), value, 0.0)

    # Slope beyond the last strike comes only from calls.
    slope_row = np.concatenate([[0.0], np.ones(n), -np.ones(n), np.zeros(2 * n)])
    tail = _tail_slope(payoff, float(grid[-1]))

    if side == "super":  # min cost s.t. pay >= target, slope >= tail
        a_ub = np.vstack([-pay, -slope_row])
        b_ub = np.concatenate([-target, [-tail]])
        res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        sign = 1.0
    else:  # max value of a dominated portfolio (what selling it raises) s.t. pay <= target
        a_ub = np.vstack([pay, slope_row])
        b_ub = np.concatenate([target, [tail]])
        res = linprog(-cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        sign = -1.0
    if res.status != 0:
        raise RuntimeError(f"Replication LP failed ({side}): {res.message}")

    x = res.x
    return Replication(
        value=float(sign * res.fun),
        bond=float(x[0]),
        calls=x[1 : n + 1] - x[n + 1 : 2 * n + 1],
        puts=x[2 * n + 1 : 3 * n + 1] - x[3 * n + 1 :],
        strikes=ks,
    )
