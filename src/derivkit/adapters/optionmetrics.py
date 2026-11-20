"""OptionMetrics IvyDB adapter (end-of-day SPX / SPXW / XSP options via WRDS).

Input is the ``opprcd`` option price table exported from WRDS (CSV or a DataFrame).
Columns used: ``date, exdate, cp_flag, strike_price, best_bid, best_offer, volume,
open_interest, optionid, symbol, am_settlement``.

Conventions handled here:
  * ``strike_price`` is the strike times 1000.
  * ``am_settlement = 1``: settles on the special opening quotation of ``exdate``;
    ``0``: settles on the 4pm close (SPXW dailies, XSP).
  * Quotes are end-of-day, stamped at 16:00 New York time on ``date``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from derivkit.adapters.base import Adapter, Tables
from derivkit.core.instruments import AssetClass, Instrument, SettlementKind, SettlementRule
from derivkit.core.payoffs import OptionType, Vanilla

NY = ZoneInfo("America/New_York")
CLOSE = time(16, 0)
OPEN = time(9, 30)

PM_RULE = SettlementRule(SettlementKind.CLOSE, "sp_dji:spx_close")
AM_RULE = SettlementRule(SettlementKind.OPEN_QUOTATION, "cboe:spx_soq")


def _ny(day: Any, at: time) -> datetime:
    d = pd.Timestamp(day).date()
    return datetime.combine(d, at, tzinfo=NY)


class OptionMetricsAdapter(Adapter):
    venue = "cboe"

    def __init__(self, underlying_id: str = "SPX", multiplier: float = 100.0):
        self.underlying_id = underlying_id
        self.multiplier = multiplier

    def normalize(self, raw: dict[str, Any]) -> Tables:
        """``raw["opprcd"]``: DataFrame (or path to CSV) of IvyDB option prices."""
        src = raw["opprcd"]
        df = pd.read_csv(src) if isinstance(src, str) else src.copy()
        df = df.drop_duplicates(subset=["optionid", "date"])

        instruments = [
            self.instrument(r) for r in df.drop_duplicates(subset=["optionid"]).to_dict("records")
        ]
        quotes = pd.DataFrame(
            {
                "ts_utc": [_ny(d, CLOSE) for d in df["date"]],
                "instrument_id": "om:" + df["optionid"].astype(str),
                "bid": df["best_bid"].astype(float),
                "ask": df["best_offer"].astype(float),
                "bid_size": float("nan"),
                "ask_size": float("nan"),
                "source": "optionmetrics:opprcd",
                "synchronised": False,
            }
        )
        quotes["ts_utc"] = pd.to_datetime(quotes["ts_utc"], utc=True)
        tables = Tables(instruments=self.instruments_frame(instruments), quotes=quotes)
        return tables.validated()

    def instrument(self, row: Mapping[Any, Any]) -> Instrument:
        am = int(row.get("am_settlement", 0) or 0) == 1
        expiry = _ny(row["exdate"], OPEN if am else CLOSE).astimezone(ZoneInfo("UTC"))
        opt = OptionType.CALL if str(row["cp_flag"]).upper() == "C" else OptionType.PUT
        return Instrument(
            instrument_id=f"om:{row['optionid']}",
            venue=self.venue,
            underlying_id=self.underlying_id,
            asset_class=AssetClass.INDEX,
            payoff=Vanilla(opt, float(row["strike_price"]) / 1000.0),
            expiry_utc=expiry,
            settlement=AM_RULE if am else PM_RULE,
            multiplier=self.multiplier,
        )
