"""The two-tier entry rule from the proposal.

Tier 1 (model-free): the target's executable price is outside the replication bounds by
more than its fee and a buffer -> trade against the bounding spread.
Tier 2 (model): inside the bounds, but the executable price beats the calibrated fair
value by more than all-in costs plus a minimum edge -> trade, hedged with the nearest spread.

Prices are per contract paying 1 (e.g. Kalshi dollars). Fees are per contract here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from derivkit.replicate.bounds import Bounds


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass(frozen=True)
class Signal:
    action: Action
    tier: int  # 1 = model-free, 2 = model, 0 = no trade
    edge: float  # expected profit per contract after fees and buffer (0 if no trade)
    price: float | None  # price we would trade at


@dataclass(frozen=True)
class Thresholds:
    """Per-contract cost allowances.

    Tier-1 bounds are already built from executable option bids and asks, so the hedge's
    half-spreads are charged only to tier-2 trades (``hedge_spread_cost``). Option
    commissions (``hedge_fees``) and the settlement-basis buffer apply to both tiers.
    """

    basis_buffer: float = 0.0
    hedge_fees: float = 0.0
    hedge_spread_cost: float = 0.0
    min_edge: float = 0.0


def evaluate(
    bid: float | None,
    ask: float | None,
    fair_value: float,
    bounds: Bounds,
    fee_per_contract: Callable[[float], float],
    thresholds: Thresholds = Thresholds(),
    allowed_tiers: frozenset[int] = frozenset({1, 2}),
) -> Signal:
    """Return the best signal for one target quote, using only ``allowed_tiers``."""
    t = thresholds
    tier1_costs = t.basis_buffer + t.hedge_fees
    tier2_costs = tier1_costs + t.hedge_spread_cost + t.min_edge
    candidates: list[Signal] = []

    if ask is not None:
        buy_cost = ask + fee_per_contract(ask)
        edge1 = bounds.lower.value - buy_cost - tier1_costs
        if edge1 > 0:
            candidates.append(Signal(Action.BUY, 1, edge1, ask))
        edge2 = fair_value - buy_cost - tier2_costs
        if edge2 > 0:
            candidates.append(Signal(Action.BUY, 2, edge2, ask))

    if bid is not None:
        sell_proceeds = bid - fee_per_contract(bid)
        edge1 = sell_proceeds - bounds.upper.value - tier1_costs
        if edge1 > 0:
            candidates.append(Signal(Action.SELL, 1, edge1, bid))
        edge2 = sell_proceeds - fair_value - tier2_costs
        if edge2 > 0:
            candidates.append(Signal(Action.SELL, 2, edge2, bid))

    candidates = [c for c in candidates if c.tier in allowed_tiers]
    if not candidates:
        return Signal(Action.NONE, 0, 0.0, None)
    # Prefer model-free trades; among equals, the larger edge.
    return min(candidates, key=lambda s: (s.tier, -s.edge))
