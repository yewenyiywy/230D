"""Pricing identities: parity, Greeks, implied vol, SVI and the smile-consistent digital."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.calibrate.svi import SVI, fit_svi
from derivkit.price import (
    black_delta,
    black_digital,
    black_gamma,
    black_price,
    black_vega,
    finite_difference_digital,
    implied_vol,
    svi_call,
    svi_digital,
)
from tests.conftest import DF, FORWARD, T

K = np.linspace(6800, 8000, 25)
SIG = 0.2


def test_put_call_parity():
    c = black_price(FORWARD, K, 0.5, SIG, DF, call=True)
    p = black_price(FORWARD, K, 0.5, SIG, DF, call=False)
    np.testing.assert_allclose(c - p, DF * (FORWARD - K), atol=1e-9)


def test_flat_digital_is_minus_dc_dk():
    h = 1e-3
    fd = finite_difference_digital(K, lambda k: black_price(FORWARD, k, 0.5, SIG, DF), h)
    np.testing.assert_allclose(fd, black_digital(FORWARD, K, 0.5, SIG, DF), atol=1e-7)
    np.testing.assert_allclose(
        black_digital(FORWARD, K, 0.5, SIG, DF)
        + black_digital(FORWARD, K, 0.5, SIG, DF, above=False),
        DF,
    )


@pytest.mark.parametrize(
    "greek, bump",
    [
        (lambda: black_vega(FORWARD, K, 0.5, SIG, DF), "sigma"),
        (lambda: black_delta(FORWARD, K, 0.5, SIG, DF), "forward"),
    ],
)
def test_first_order_greeks_match_finite_differences(greek, bump):
    h = 1e-5 if bump == "sigma" else 1e-3
    if bump == "sigma":
        up, dn = (
            black_price(FORWARD, K, 0.5, SIG + h, DF),
            black_price(FORWARD, K, 0.5, SIG - h, DF),
        )
    else:
        up, dn = (
            black_price(FORWARD + h, K, 0.5, SIG, DF),
            black_price(FORWARD - h, K, 0.5, SIG, DF),
        )
    np.testing.assert_allclose(greek(), (up - dn) / (2 * h), rtol=1e-5, atol=1e-6)


def test_gamma_matches_finite_difference():
    h = 0.5
    fd = (
        black_price(FORWARD + h, K, 0.5, SIG, DF)
        - 2 * black_price(FORWARD, K, 0.5, SIG, DF)
        + black_price(FORWARD - h, K, 0.5, SIG, DF)
    ) / h**2
    np.testing.assert_allclose(black_gamma(FORWARD, K, 0.5, SIG, DF), fd, rtol=1e-4, atol=1e-9)


def test_expiry_returns_intrinsic():
    assert black_price(100, 90, 0.0, 0.2) == pytest.approx(10)
    assert black_digital(100, 90, 0.0, 0.2) == 1.0
    assert black_digital(100, 110, 0.0, 0.2) == 0.0


def test_implied_vol_round_trip():
    vols = np.linspace(0.05, 0.9, 25)
    prices = black_price(FORWARD, K, 0.5, vols, DF, call=True)
    np.testing.assert_allclose(implied_vol(prices, FORWARD, K, 0.5, DF, call=True), vols, atol=1e-8)


def test_implied_vol_nan_outside_bounds():
    assert np.isnan(implied_vol(0.0, 100, 90, 0.5))
    assert np.isnan(implied_vol(200.0, 100, 90, 0.5))


def test_svi_fit_recovers_smile(skewed_svi: SVI):
    k = np.linspace(-0.03, 0.03, 31)
    fitted = fit_svi(k, skewed_svi.vol(k), T)
    np.testing.assert_allclose(fitted.vol(k), skewed_svi.vol(k), atol=2e-4)
    assert fitted.is_butterfly_free(np.linspace(-0.05, 0.05, 201))


def test_smile_digital_equals_minus_dc_dk_on_the_smile(skewed_svi: SVI):
    strikes = np.linspace(7300, 7500, 21)
    fd = finite_difference_digital(strikes, lambda k: svi_call(k, FORWARD, skewed_svi, DF), 1e-3)
    np.testing.assert_allclose(svi_digital(strikes, FORWARD, skewed_svi, DF), fd, atol=1e-6)


def test_negative_skew_makes_upside_digitals_dearer(skewed_svi: SVI):
    strikes = np.array([7420.0, 7450.0])
    flat = black_digital(FORWARD, strikes, T, skewed_svi.vol_at_strike(strikes, FORWARD), DF)
    assert np.all(svi_digital(strikes, FORWARD, skewed_svi, DF) > flat)


def test_bucket_prices_sum_to_discount_factor(skewed_svi: SVI):
    edges = np.arange(7000, 7801, 25.0)
    above = svi_digital(edges, FORWARD, skewed_svi, DF)
    buckets = np.concatenate([[DF - above[0]], above[:-1] - above[1:], [above[-1]]])
    assert buckets.sum() == pytest.approx(DF, abs=1e-12)
    assert np.all(buckets >= -1e-12)
