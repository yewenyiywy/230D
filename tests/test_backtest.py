"""End-to-end backtest on the synthetic market."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivkit.backtest import ATTRIBUTION, Settings, run
from derivkit.calibrate.svi import SVI
from derivkit.cli import main
from derivkit.config import load_pair
from derivkit.core.schemas import validate
from derivkit.evaluate import block_bootstrap_mean, max_drawdown, summarise, walk_forward
from derivkit.market import build_state, parity_forward
from tests import synthetic

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_config(load_pair(ROOT / "configs/pairs/spx_daily_digitals.yaml"))


@pytest.fixture(scope="module")
def market() -> synthetic.Market:
    return synthetic.build()


def test_synthetic_market_passes_schemas(market):
    validate(market.instruments, "instruments")
    validate(market.quotes, "quotes")
    validate(market.settlements, "settlements")


def test_parity_recovers_forward_and_discount():
    m = synthetic.build(n_days=1)
    from derivkit.replicate.chain import OptionChain

    hedge = m.instruments[m.instruments["venue"] == "cboe"]
    frame = m.quotes.merge(hedge, on="instrument_id")
    chain = OptionChain.from_frame(frame)
    f, df = parity_forward(chain)
    assert f == pytest.approx(7400.0, abs=1e-6)
    assert df == pytest.approx(np.exp(-0.04 / 365), abs=1e-9)
    t = 1 / 365
    state = build_state(chain, t)
    true = SVI(a=0.5 * 0.15**2 * t, b=0.003 * t * 252, rho=-0.6, m=0.0, s=0.01, t=t)
    strikes = np.array([7300.0, 7400.0, 7500.0])
    np.testing.assert_allclose(
        state.svi.vol_at_strike(strikes, f), true.vol_at_strike(strikes, 7400.0), atol=2e-3
    )


def test_only_mispriced_contracts_trade(market, settings):
    res = run(market.instruments, market.quotes, market.settlements, settings)
    assert set(res.trades["instrument_id"]) == set(market.cheap)
    assert (res.trades["tier"] == 1).all() and (res.trades["action"] == "buy").all()
    assert res.skipped["no_signal"] > 0


def test_attribution_adds_up_to_pnl(market, settings):
    trades = run(market.instruments, market.quotes, market.settlements, settings).settled
    assert not trades.empty
    np.testing.assert_allclose(trades[ATTRIBUTION].sum(axis=1), trades["pnl"], atol=1e-12)


@pytest.mark.parametrize("move", np.linspace(-0.01, 0.01, 21))
def test_tier1_trade_cannot_lose_more_than_its_expected_edge_says(settings, move):
    m = synthetic.build(n_days=1, cheap_days=(0,), settle_moves=[float(move)])
    t = run(m.instruments, m.quotes, m.settlements, settings).settled
    assert len(t) == 1
    row = t.iloc[0]
    # Long digital, short the sub-replicating spread: worst case is the locked-in edge.
    assert row["pnl"] >= row["expected_edge"] - 1e-9
    assert row["expected_edge"] > 0


def test_no_look_ahead(market, settings):
    late = market.quotes.copy()
    is_target = late["instrument_id"].str.startswith("kalshi:")
    late.loc[is_target, "ts_utc"] += pd.Timedelta(minutes=1)
    res = run(market.instruments, late, market.settlements, settings)
    assert res.trades.empty
    assert res.skipped["no_target_quote"] > 0


def test_unsettled_trades_are_kept_but_flagged(market, settings):
    res = run(market.instruments, market.quotes, market.settlements.iloc[:0], settings)
    assert not res.trades.empty and not res.trades["settled"].any()


def test_metrics(market, settings):
    trades = run(market.instruments, market.quotes, market.settlements, settings).trades
    s = summarise(trades, n_boot=500)
    assert s.n_trades == len(market.cheap) and s.n_tier1 == s.n_trades
    assert s.total_pnl == pytest.approx(trades["pnl"].sum())
    assert sum(s.attribution.values()) == pytest.approx(s.total_pnl)
    lo, hi = block_bootstrap_mean(trades, n_boot=500)
    assert lo <= s.mean_pnl_per_trade <= hi
    assert max_drawdown(pd.Series([1.0, -2.0, 0.5, 3.0])) == pytest.approx(2.0)


def test_walk_forward_keeps_holdout_last_and_untouched():
    days = [date(2026, 1, d) for d in range(1, 21)]
    plan = walk_forward(days, n_folds=3, holdout_fraction=0.2)
    assert plan.holdout == tuple(days[16:])
    assert all(max(f.train) < min(f.test) for f in plan.folds)
    assert all(max(f.test) < plan.holdout[0] for f in plan.folds)
    assert set(plan.folds[-1].train) | set(plan.folds[-1].test) == set(plan.development)


def test_cli_runs_development_period_only(tmp_path: Path, market):
    data = tmp_path / "data"
    data.mkdir()
    market.instruments.to_csv(data / "instruments.csv", index=False)
    market.quotes.to_csv(data / "quotes.csv", index=False)
    market.settlements.to_csv(data / "settlements.csv", index=False)
    out = tmp_path / "out"
    cfg = str(ROOT / "configs/pairs/spx_daily_digitals.yaml")
    assert main(["backtest", cfg, "--data", str(data), "--out", str(out)]) == 0
    report = json.loads((out / "summary_development.json").read_text())
    trades = pd.read_csv(out / "trades_development.csv")
    holdout_start = pd.Timestamp(report["holdout_start"]).date()
    decided = (
        pd.to_datetime(trades["decision_ts"], utc=True).dt.tz_convert("America/New_York").dt.date
    )
    assert (decided < holdout_start).all()

    assert main(["backtest", cfg, "--data", str(data), "--out", str(out), "--open-holdout"]) == 0
    assert (out / "summary_holdout.json").exists()


def test_measured_basis_requires_an_estimate():
    cfg = load_pair(ROOT / "configs/pairs/btc_0800_digitals.yaml")
    with pytest.raises(ValueError, match="measured"):
        Settings.from_config(cfg)
    assert Settings.from_config(cfg, basis_buffer=0.01).thresholds.basis_buffer == 0.01
