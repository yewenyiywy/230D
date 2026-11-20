from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from derivkit.config import load_pair
from derivkit.costs import KalshiFee, ceil_cent, fee_in_force
from derivkit.replicate.bounds import Bound, Bounds
from derivkit.strategy import Action, Thresholds, evaluate

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "contracts, price, expected",
    [
        # Rows from Kalshi's published general fee table.
        (1, 0.01, 0.01),
        (100, 0.01, 0.07),
        (100, 0.05, 0.34),
        (100, 0.10, 0.63),
        (100, 0.15, 0.90),
        (1, 0.20, 0.02),
        (100, 0.20, 1.12),
        (100, 0.25, 1.32),
        (100, 0.55, 1.74),
    ],
)
def test_kalshi_taker_fee_matches_published_table(contracts, price, expected):
    assert KalshiFee(date(2026, 1, 1)).fee(contracts, price) == pytest.approx(expected)


def test_fee_helpers():
    assert ceil_cent(1.75) == 1.75
    assert ceil_cent(1.7501) == 1.76
    half = KalshiFee(date(2026, 1, 1), multiplier=0.5)
    assert half.fee(100, 0.5) == pytest.approx(0.88)
    schedule = [KalshiFee(date(2025, 1, 1)), half]
    assert fee_in_force(schedule, date(2025, 6, 1)).effective_from == date(2025, 1, 1)
    assert fee_in_force(schedule, date(2026, 6, 1)) is half
    with pytest.raises(LookupError):
        fee_in_force(schedule, date(2024, 1, 1))


BOUNDS = Bounds(Bound(0.55, 7400, 7405), Bound(0.65, 7395, 7400))
NO_FEE = lambda p: 0.0  # noqa: E731


def test_tier1_buy_below_lower_bound():
    sig = evaluate(bid=0.48, ask=0.50, fair_value=0.60, bounds=BOUNDS, fee_per_contract=NO_FEE)
    assert (sig.action, sig.tier) == (Action.BUY, 1)
    assert sig.edge == pytest.approx(0.05)


def test_tier2_sell_above_fair_value_inside_bounds():
    sig = evaluate(bid=0.63, ask=0.66, fair_value=0.60, bounds=BOUNDS, fee_per_contract=NO_FEE)
    assert (sig.action, sig.tier) == (Action.SELL, 2)
    assert sig.edge == pytest.approx(0.03)


def test_costs_remove_the_edge():
    fee = lambda p: 0.02  # noqa: E731
    sig = evaluate(
        bid=0.61,
        ask=0.62,
        fair_value=0.60,
        bounds=BOUNDS,
        fee_per_contract=fee,
        thresholds=Thresholds(hedge_cost=0.005),
    )
    assert sig.action is Action.NONE


@pytest.mark.parametrize("name", ["spx_daily_digitals", "btc_0800_digitals"])
def test_pair_configs_load(name):
    cfg = load_pair(ROOT / "configs" / "pairs" / f"{name}.yaml")
    assert cfg.pair == name
    assert cfg.target.adapter == "kalshi"
