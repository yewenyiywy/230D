# derivkit: replication-based derivatives pricing and trading

MFE 230D project · Lex Qiu · Wenyi Ye · Aditya Aryan

We price a contract quoted on one venue from the portfolio of listed options that
replicates it, and trade only when the quote is outside cost-adjusted bounds, holding
the hedge. The first test case is Kalshi's daily S&P 500 and BTC contracts (cash-or-nothing
digitals) against SPXW and Deribit options. The code itself is venue- and asset-agnostic.

## How it is organised

```
adapters  ->  standard tables  ->  calibrate  ->  price  ->  replicate  ->  strategy  ->  backtest  ->  evaluate
(venue-specific)                   (everything from here on is the same for every venue)
```

Only `adapters/` and the YAML files in `configs/pairs/` know which venue or asset class is
in use. Moving to a new venue means writing one adapter that passes
`tests/test_adapters.py::test_contract`, plus one pair config.

| Package | What it does | Status |
| --- | --- | --- |
| `core` | `Instrument`, typed payoffs (vanilla, digital, range, spread), settlement rules, the seven standard table schemas and their validator | done |
| `adapters` | Kalshi, Deribit, OptionMetrics (WRDS export), CSV; network fetchers plus pure `normalize` | done (live fetchers untested in CI) |
| `calibrate` | Raw SVI fit per expiry, with Gatheral's butterfly check | done |
| `price` | Black-76 prices and Greeks, implied vol, smile-consistent digital | done |
| `replicate` | Model-free digital/range bounds from call spreads; general super/sub-replication LP | done |
| `costs` | Kalshi fee formula by date; Deribit and per-contract fee models | done (Deribit parameters to verify) |
| `strategy` | Two-tier entry rule: outside bounds (tier 1) or beyond model fair value (tier 2) | done |
| `config` | Pair config schema and loader | done |
| `hedge`, `backtest`, `evaluate` | Hedging, walk-forward backtest with P&L attribution, holdout metrics | next |
| Monte Carlo, PDE, Deep BSDE engines | Pricing for path-dependent and multi-asset payoffs | later |

## Getting started

```bash
uv sync                  # Python 3.11+
uv run pytest            # all tests, no network
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

Pricing a Kalshi-style digital and bounding it with listed calls:

```python
import numpy as np
from derivkit.calibrate import fit_svi
from derivkit.price import implied_vol, svi_digital
from derivkit.replicate import OptionChain, digital_above_bounds

# chain: one expiry's call quotes (per unit of underlying), forward F, time t, discount df
iv = implied_vol((chain.call_bid + chain.call_ask) / 2, F, chain.strikes, t, df)
ok = np.isfinite(iv)
svi = fit_svi(np.log(chain.strikes[ok] / F), iv[ok], t)

fair = svi_digital(7399.9999, F, svi, df)  # "7,400 or above"
bounds = digital_above_bounds(chain, 7399.9999, tick=0.01)  # S&P prints to the cent
print(bounds.lower.value, fair, bounds.upper.value)
```

## Conventions

* Timestamps are timezone-aware UTC; prices are USD per contract; strikes are in
  underlying units. Option chains passed to `replicate` are per unit of underlying
  (divide by the multiplier).
* Raw API responses are stored unchanged; adapters only normalise.
* Data, MLflow runs and secrets never go in git (see `.gitignore`).

## CI

`.github/workflows/ci.yml` runs lint, format check, type check and the test suite on every
pull request. Tests use the synthetic fixtures in `tests/fixtures/`; tests that call live
APIs are marked `network` and skipped in CI.
