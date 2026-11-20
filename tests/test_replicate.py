"""Replication bounds: they must contain the model price and dominate the payoff."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.calibrate.svi import SVI
from derivkit.core.payoffs import Digital, OptionType, Range, Vanilla, payoff_value
from derivkit.price import svi_digital
from derivkit.replicate import digital_above_bounds, payoff_bounds, replicate
from derivkit.replicate.chain import OptionChain
from tests.conftest import DF, FORWARD, make_chain


@pytest.mark.parametrize("strike", [7372.5, 7400.0, 7417.3, 7455.0])
def test_bounds_contain_model_digital(chain: OptionChain, skewed_svi: SVI, strike: float):
    b = digital_above_bounds(chain, strike)
    model = float(svi_digital(strike, FORWARD, skewed_svi, DF))
    assert b.contains(model)
    assert 0.0 <= b.lower.value <= b.upper.value <= DF


def test_kalshi_style_strike_uses_the_listed_strike_on_a_cent_grid(chain: OptionChain):
    # "7,400 or above" is Digital(7399.9999): on a cent grid, S > 7399.9999 iff S >= 7400.
    b = digital_above_bounds(chain, 7399.9999, tick=0.01)
    assert b.upper.k2 == 7400.0
    assert b.lower.k1 == 7400.0
    continuous = digital_above_bounds(chain, 7399.9999, tick=0.0)
    assert continuous.upper.k2 == 7395.0  # looser without the tick
    assert b.width < continuous.width


def test_bounds_tighten_with_tighter_markets(skewed_svi: SVI):
    wide = digital_above_bounds(make_chain(skewed_svi, half_spread=0.2), 7410.0)
    tight = digital_above_bounds(make_chain(skewed_svi, half_spread=0.01), 7410.0)
    assert tight.width < wide.width


def test_below_and_range_bounds_contain_model(chain: OptionChain, skewed_svi: SVI):
    up = lambda k: float(svi_digital(k, FORWARD, skewed_svi, DF))  # noqa: E731
    below = payoff_bounds(chain, Digital(7390.0, above=False), tick=0.01)
    assert below.contains(DF - up(7390.0))
    rng = payoff_bounds(chain, Range(7400.0, 7424.9999), tick=0.01)
    assert rng.contains(up(7400.0) - up(7424.9999))
    open_low = payoff_bounds(chain, Range(None, 7424.9999), tick=0.01)
    assert open_low.contains(DF - up(7424.9999))


def test_missing_quotes_fall_back_to_trivial_bounds():
    ks = np.array([7400.0, 7405.0])
    empty = OptionChain(ks, np.full(2, np.nan), np.full(2, np.nan), df=DF)
    b = digital_above_bounds(empty, 7402.0)
    assert (b.lower.value, b.upper.value) == (0.0, DF)


@pytest.mark.parametrize(
    "payoff",
    [Digital(7412.0), Range(7400.0, 7425.0), Vanilla(OptionType.CALL, 7400.0)],
)
def test_lp_portfolios_dominate_and_bracket(chain: OptionChain, payoff):
    sup = replicate(chain, payoff, "super")
    sub = replicate(chain, payoff, "sub")
    s = np.linspace(0, 2 * chain.strikes.max(), 20001)
    target = payoff_value(payoff, s)
    assert np.all(sup.payoff(s) >= target - 1e-7)
    assert np.all(sub.payoff(s) <= target + 1e-7)
    assert sub.value <= sup.value + 1e-9


def test_lp_is_at_least_as_tight_as_pairwise(chain: OptionChain):
    pair = digital_above_bounds(chain, 7412.0)
    assert replicate(chain, Digital(7412.0), "super").value <= pair.upper.value + 1e-9
    assert replicate(chain, Digital(7412.0), "sub").value >= pair.lower.value - 1e-9


def test_lp_prices_a_listed_call_within_its_own_quote(chain: OptionChain):
    i = int(np.searchsorted(chain.strikes, 7400.0))
    call = Vanilla(OptionType.CALL, 7400.0)
    assert replicate(chain, call, "super").value <= chain.call_ask[i] + 1e-9
    assert replicate(chain, call, "sub").value >= chain.call_bid[i] - 1e-9
