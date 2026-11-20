from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from derivkit.core import (
    AssetClass,
    Digital,
    Instrument,
    OptionType,
    Range,
    SchemaError,
    SettlementKind,
    SettlementRule,
    Spread,
    Vanilla,
    payoff_value,
    validate,
)


def test_payoffs():
    s = np.array([7399.99, 7400.0, 7412.0, 7425.0])
    assert payoff_value(Digital(7399.9999), s).tolist() == [0, 1, 1, 1]
    assert payoff_value(Digital(7400, above=False), s).tolist() == [1, 0, 0, 0]
    assert payoff_value(Range(7400, 7424.9999), s).tolist() == [0, 1, 1, 0]
    assert payoff_value(Vanilla(OptionType.CALL, 7400), s).tolist() == pytest.approx([0, 0, 12, 25])
    spread = Spread(((1.0, Vanilla(OptionType.CALL, 7400)), (-1.0, Vanilla(OptionType.CALL, 7410))))
    assert payoff_value(spread, s).tolist() == pytest.approx([0, 0, 10, 10])


def test_range_rejects_two_open_sides():
    with pytest.raises(ValueError):
        Range(None, None)


def test_instrument_requires_utc_aware_expiry():
    rule = SettlementRule(SettlementKind.CLOSE, "x")
    with pytest.raises(ValueError):
        Instrument("a", "v", "SPX", AssetClass.INDEX, Digital(1.0), datetime(2026, 1, 1), rule)
    inst = Instrument(
        "a", "v", "SPX", AssetClass.INDEX, Range(1, 2), datetime(2026, 1, 1, tzinfo=UTC), rule
    )
    rec = inst.to_record()
    assert rec["payoff_type"] == "range" and rec["strike"] == 1 and rec["strike_upper"] == 2


def _quotes(**overrides) -> pd.DataFrame:
    row = {
        "ts_utc": pd.Timestamp("2026-11-20T15:00Z"),
        "instrument_id": "x",
        "bid": 0.4,
        "ask": 0.5,
        "bid_size": np.nan,
        "ask_size": np.nan,
        "source": "test",
        "synchronised": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_accepts_good_quotes():
    assert len(validate(_quotes(), "quotes")) == 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"bid": 0.6}, "crossed"),
        ({"ts_utc": pd.Timestamp("2026-11-20T15:00")}, "UTC"),
        ({"instrument_id": None}, "null"),
    ],
)
def test_validate_rejects_bad_quotes(overrides, message):
    with pytest.raises(SchemaError, match=message):
        validate(_quotes(**overrides), "quotes")


def test_validate_reports_missing_columns():
    with pytest.raises(SchemaError, match="missing columns"):
        validate(_quotes().drop(columns=["source"]), "quotes")
