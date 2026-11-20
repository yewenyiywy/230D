"""Adapter tests. ``test_contract`` runs against every adapter: a new venue is done when
it passes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from derivkit.adapters import CsvAdapter, DeribitAdapter, KalshiAdapter, OptionMetricsAdapter
from derivkit.adapters.base import Tables
from derivkit.adapters.deribit import parse_name

FIX = Path(__file__).parent / "fixtures"


def kalshi_tables() -> Tables:
    return KalshiAdapter().normalize(json.loads((FIX / "kalshi.json").read_text()))


def deribit_tables() -> Tables:
    return DeribitAdapter().normalize(json.loads((FIX / "deribit.json").read_text()))


def optionmetrics_tables() -> Tables:
    return OptionMetricsAdapter().normalize({"opprcd": str(FIX / "opprcd.csv")})


def csv_tables(tmp_path: Path) -> Tables:
    src = kalshi_tables()
    for name in ("instruments", "quotes", "trades", "settlements"):
        getattr(src, name).to_csv(tmp_path / f"{name}.csv", index=False)
    return CsvAdapter().normalize({"directory": str(tmp_path)})


@pytest.mark.parametrize("make", ["kalshi", "deribit", "optionmetrics", "csv"])
def test_contract(make: str, tmp_path: Path):
    tables = {
        "kalshi": kalshi_tables,
        "deribit": deribit_tables,
        "optionmetrics": optionmetrics_tables,
        "csv": lambda: csv_tables(tmp_path),
    }[make]()
    assert not tables.instruments.empty
    assert str(tables.instruments["expiry_utc"].dt.tz) == "UTC"
    known = set(tables.instruments["instrument_id"])
    for name in ("quotes", "trades", "settlements"):
        df: pd.DataFrame = getattr(tables, name)
        if not df.empty:
            assert set(df["instrument_id"]) <= known, f"{name} refers to unknown instruments"
    if not tables.quotes.empty:
        assert str(tables.quotes["ts_utc"].dt.tz) == "UTC"


def test_kalshi_payoffs_and_prices():
    t = kalshi_tables()
    inst = t.instruments.set_index("instrument_id")
    above = inst.loc["kalshi:KXINXU-26NOV20H1600-T7399.9999"]
    assert above["payoff_type"] == "digital_above" and above["strike"] == pytest.approx(7399.9999)
    bucket = inst.loc["kalshi:KXINX-26NOV20H1600-B7412"]
    assert (bucket["strike"], bucket["strike_upper"]) == (7400, 7424.9999)
    btc = inst.loc["kalshi:KXBTCD-26NOV2003-T91999.99"]
    assert btc["settlement_rule"].startswith("trimmed_mean:60s:0.2")
    assert btc["expiry_utc"] == pd.Timestamp("2026-11-20T08:00Z")

    # Dollar strings and legacy cents parse to the same units.
    assert t.quotes["bid"].tolist() == pytest.approx([0.61, 0.62])
    assert t.quotes["ask"].tolist() == pytest.approx([0.64, 0.65])
    assert t.trades.set_index("trade_id").loc["a1b2", "price"] == pytest.approx(0.62)
    assert t.trades.set_index("trade_id").loc["c3d4", "aggressor_side"] == "sell"
    # Only finalised markets produce settlements.
    assert len(t.settlements) == 2


def test_deribit_names_and_usd_conversion():
    p = parse_name("BTC-21NOV26-92000-C")
    assert p["expiry"] == pd.Timestamp("2026-11-21T08:00Z").to_pydatetime()
    assert p["strike"] == 92000 and not p["linear"]
    assert parse_name("BTC_USDC-21NOV26-90000-P")["linear"]

    t = deribit_tables()
    q = t.quotes.set_index("instrument_id")
    # Inverse: BTC premium x index; linear: already USD.
    assert q.loc["deribit:BTC-21NOV26-92000-C", "bid"] == pytest.approx(0.0105 * 92000)
    assert pd.isna(q.loc["deribit:BTC-21NOV26-93000-C", "ask"])
    assert q.loc["deribit:BTC_USDC-21NOV26-90000-P", "ask"] == pytest.approx(430.0)
    assert "deribit:BTC-PERPETUAL" not in q.index

    s = t.settlements.set_index("instrument_id")
    assert s.loc["deribit:BTC-21NOV26-92000-C", "settlement_value"] == pytest.approx(500.0)
    assert s.loc["deribit:BTC_USDC-21NOV26-90000-P", "settlement_value"] == 0.0


def test_optionmetrics_strikes_and_settlement_style():
    t = optionmetrics_tables()
    inst = t.instruments.set_index("instrument_id")
    assert inst.loc["om:9001", "strike"] == 7400.0
    assert inst.loc["om:9001", "settlement_rule"] == "close:sp_dji:spx_close"
    assert inst.loc["om:9001", "expiry_utc"] == pd.Timestamp("2026-11-20T21:00Z")
    assert inst.loc["om:9004", "settlement_rule"].startswith("open_quotation")
    assert inst.loc["om:9004", "expiry_utc"] == pd.Timestamp("2026-12-18T14:30Z")
    assert t.quotes["ts_utc"].iloc[0] == pd.Timestamp("2026-11-19T21:00Z")
