"""A small synthetic market in the standard tables, for end-to-end backtest tests.

Each day: an SPXW-like call/put chain at 16:00 New York for the next day's close, priced
off a skewed SVI smile with a fixed half-spread, and Kalshi-like "above" digitals on the
same close. Every contract is quoted near its model value except, on chosen days, one
contract whose ask sits below the model-free lower bound (a tier-1 opportunity).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from derivkit.calibrate.svi import SVI
from derivkit.core.payoffs import Digital
from derivkit.price.black import black_price
from derivkit.price.smile_digital import svi_digital
from derivkit.replicate.bounds import payoff_bounds
from derivkit.replicate.chain import OptionChain

NY = ZoneInfo("America/New_York")
MULT = 100.0
STEP = 5.0
HALF_SPREAD = 0.05  # per unit of underlying
RATE = 0.04


@dataclass
class Market:
    instruments: pd.DataFrame
    quotes: pd.DataFrame
    settlements: pd.DataFrame
    cheap: list[str]  # target ids quoted below their lower bound


def _ts(day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time(16, 0), tzinfo=NY)).tz_convert("UTC")


def build(
    n_days: int = 6,
    cheap_days: tuple[int, ...] = (0, 3),
    settle_moves: list[float] | None = None,
    seed: int = 7,
    target_offsets: tuple[float, ...] = (-20.0, 0.0, 20.0),
) -> Market:
    rng = np.random.default_rng(seed)
    start = date(2026, 11, 2)
    days = [start + timedelta(days=i) for i in range(n_days + 1)]
    inst, quotes, settles, cheap = [], [], [], []
    forward = 7400.0
    for i in range(n_days):
        today, tomorrow = days[i], days[i + 1]
        ts, expiry = _ts(today), _ts(tomorrow)
        t = (expiry - ts).total_seconds() / (365 * 24 * 3600)
        df = float(np.exp(-RATE * t))
        svi = SVI(a=0.5 * 0.15**2 * t, b=0.003 * t * 252, rho=-0.6, m=0.0, s=0.01, t=t)
        strikes = forward + STEP * np.arange(-40, 41)
        vol = svi.vol_at_strike(strikes, forward)
        calls = black_price(forward, strikes, t, vol, df, call=True)
        puts = black_price(forward, strikes, t, vol, df, call=False)
        tag = tomorrow.strftime("%y%m%d")
        for k, c, p in zip(strikes, calls, puts, strict=True):
            for kind, mid in (("call", c), ("put", p)):
                iid = f"cboe:SPXW{tag}{kind[0].upper()}{k:g}"
                inst.append(_row(iid, "cboe", kind, k, expiry, "close:sp_dji:spx_close", MULT))
                quotes.append(
                    _quote(ts, iid, max(mid - HALF_SPREAD, 0.0) * MULT, (mid + HALF_SPREAD) * MULT)
                )

        chain = OptionChain(
            strikes,
            np.maximum(calls - HALF_SPREAD, 0.0),
            calls + HALF_SPREAD,
            df=df,
        )
        move = settle_moves[i] if settle_moves else float(rng.normal(0, 0.008))
        s_t = round(forward * np.exp(move), 2)
        for j, off in enumerate(target_offsets):
            k = forward + off - 0.0001
            iid = f"kalshi:KXINXU-{tomorrow:%y%b%d}H1600-T{k:.4f}".upper().replace(
                "KALSHI", "kalshi"
            )
            inst.append(
                _row(iid, "kalshi", "digital_above", k, expiry, "close:kalshi:sp500_close", 1.0)
            )
            fair = float(svi_digital(k, forward, svi, df))
            if i in cheap_days and j == len(target_offsets) // 2:
                lower = payoff_bounds(chain, Digital(k), tick=0.01).lower.value
                bid, ask = round(lower - 0.08, 2), round(lower - 0.06, 2)
                cheap.append(iid)
            else:
                bid, ask = round(fair - 0.01, 2), round(fair + 0.01, 2)
            quotes.append(_quote(ts, iid, max(bid, 0.0), min(ask, 1.0)))
            settles.append(
                {
                    "instrument_id": iid,
                    "settlement_value": float(s_t > k),
                    "underlying_settlement_price": s_t,
                    "source": "synthetic",
                }
            )
        forward = s_t
    instruments = pd.DataFrame(inst)
    instruments["expiry_utc"] = pd.to_datetime(instruments["expiry_utc"], utc=True)
    q = pd.DataFrame(quotes)
    q["ts_utc"] = pd.to_datetime(q["ts_utc"], utc=True)
    return Market(instruments, q, pd.DataFrame(settles), cheap)


def _row(iid, venue, kind, k, expiry, rule, mult) -> dict:
    return {
        "instrument_id": iid,
        "venue": venue,
        "underlying_id": "SPX",
        "asset_class": "index",
        "payoff_type": kind,
        "strike": float(k),
        "strike_upper": np.nan,
        "expiry_utc": expiry,
        "settlement_rule": rule,
        "multiplier": mult,
        "currency": "USD",
    }


def _quote(ts, iid, bid, ask) -> dict:
    return {
        "ts_utc": ts,
        "instrument_id": iid,
        "bid": float(bid),
        "ask": float(ask),
        "bid_size": np.nan,
        "ask_size": np.nan,
        "source": "synthetic",
        "synchronised": True,
    }
