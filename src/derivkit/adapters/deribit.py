"""Deribit adapter (BTC options).

Live:    https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
History: https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time

Inverse options (``BTC-29NOV26-90000-C``) are quoted in BTC; their USD value is the BTC
premium times the index price at that moment, because at expiry the BTC payoff
max(S - K, 0) / S is worth max(S - K, 0) dollars. Linear ``BTC_USDC-...`` options are
already quoted in USD. Options expire at 08:00 UTC and settle on a TWAP of the Deribit
index from 07:30 to 08:00 UTC.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from derivkit.adapters.base import Adapter, Tables, utc
from derivkit.core.instruments import AssetClass, Instrument, SettlementKind, SettlementRule
from derivkit.core.payoffs import OptionType, Vanilla, payoff_value
from derivkit.core.schemas import empty

LIVE_URL = "https://www.deribit.com/api/v2"
HISTORY_URL = "https://history.deribit.com/api/v2"

SETTLEMENT = SettlementRule(SettlementKind.TWAP, "deribit:btc_usd_index", window_seconds=1800)

_NAME = re.compile(
    r"^(?P<base>[A-Z]+)(?P<linear>_USDC)?-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})"
    r"-(?P<strike>\d+(?:d\d+)?)-(?P<cp>[CP])$"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}


def parse_name(name: str) -> dict[str, Any]:
    """Split an option name into base, linear flag, expiry (08:00 UTC), strike and type."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"Not a Deribit option name: {name}")
    expiry = datetime(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["day"]), 8, 0, tzinfo=UTC)
    return {
        "base": m["base"],
        "linear": m["linear"] is not None,
        "expiry": expiry,
        "strike": float(m["strike"].replace("d", ".")),
        "option_type": OptionType.CALL if m["cp"] == "C" else OptionType.PUT,
    }


class DeribitAdapter(Adapter):
    venue = "deribit"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    # ---------------------------------------------------------------- fetch (network)
    def fetch_book_summary(self, currency: str = "BTC") -> list[dict[str, Any]]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{LIVE_URL}/public/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
            )
            resp.raise_for_status()
            return resp.json()["result"]

    def fetch_trades(
        self, start: datetime, end: datetime, currency: str = "BTC", count: int = 1000
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        with httpx.Client(timeout=self.timeout) as client:
            while True:
                resp = client.get(
                    f"{HISTORY_URL}/public/get_last_trades_by_currency_and_time",
                    params={
                        "currency": currency,
                        "kind": "option",
                        "start_timestamp": start_ms,
                        "end_timestamp": end_ms,
                        "count": count,
                        "sorting": "asc",
                    },
                )
                resp.raise_for_status()
                result = resp.json()["result"]
                trades = result.get("trades", [])
                out.extend(trades)
                if not result.get("has_more") or not trades:
                    return out
                start_ms = trades[-1]["timestamp"] + 1

    # ------------------------------------------------------------- normalize (pure)
    def normalize(self, raw: dict[str, Any]) -> Tables:
        """``raw`` keys: ``book_summary`` (list), ``trades`` (list), ``deliveries`` (list of
        ``{"instrument_name", "delivery_price"}``)."""
        tables = Tables()
        names: set[str] = set()
        for key in ("book_summary", "trades", "deliveries"):
            names.update(row["instrument_name"] for row in raw.get(key, []))
        names = {n for n in names if _NAME.match(n)}
        if names:
            tables.instruments = self.instruments_frame([self.instrument(n) for n in sorted(names)])
        if raw.get("book_summary"):
            tables.quotes, tables.underlyings = self._book(raw["book_summary"])
        if raw.get("trades"):
            tables.trades = self._trades(raw["trades"])
        if raw.get("deliveries"):
            tables.settlements = self._settlements(raw["deliveries"])
        return tables.validated()

    def instrument(self, name: str) -> Instrument:
        p = parse_name(name)
        return Instrument(
            instrument_id=f"deribit:{name}",
            venue=self.venue,
            underlying_id=p["base"],
            asset_class=AssetClass.CRYPTO,
            payoff=Vanilla(p["option_type"], p["strike"]),
            expiry_utc=p["expiry"],
            settlement=SETTLEMENT,
            multiplier=1.0,
        )

    def _book(self, rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        quotes, unders = [], []
        for r in rows:
            name = r["instrument_name"]
            if not _NAME.match(name):
                continue
            index = r.get("estimated_delivery_price")
            to_usd = 1.0 if parse_name(name)["linear"] else (index or float("nan"))
            bid, ask = r.get("bid_price"), r.get("ask_price")
            quotes.append(
                {
                    "ts_utc": r["creation_timestamp"],
                    "instrument_id": f"deribit:{name}",
                    "bid": bid * to_usd if bid is not None else None,
                    "ask": ask * to_usd if ask is not None else None,
                    "bid_size": None,
                    "ask_size": None,
                    "source": "deribit:book_summary",
                    "synchronised": True,
                }
            )
            unders.append(
                {
                    "ts_utc": r["creation_timestamp"],
                    "underlying_id": parse_name(name)["base"],
                    "spot": index,
                    "forward": r.get("underlying_price"),
                    "index_value": index,
                    "source": f"deribit:{r.get('underlying_index', 'index')}",
                }
            )
        q = pd.DataFrame(quotes)
        q["ts_utc"] = utc(q["ts_utc"], unit="ms")
        for col in ("bid", "ask", "bid_size", "ask_size"):
            q[col] = q[col].astype(float)
        u = pd.DataFrame(unders)
        u["ts_utc"] = utc(u["ts_utc"], unit="ms")
        u = u.drop_duplicates(subset=["ts_utc", "underlying_id", "source"])
        return q, u

    def _trades(self, trades: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for t in trades:
            name = t["instrument_name"]
            if not _NAME.match(name):
                continue
            to_usd = 1.0 if parse_name(name)["linear"] else t["index_price"]
            rows.append(
                {
                    "ts_utc": t["timestamp"],
                    "instrument_id": f"deribit:{name}",
                    "price": t["price"] * to_usd,
                    "size": float(t["amount"]),
                    "aggressor_side": t.get("direction"),
                    "trade_id": str(t["trade_id"]),
                }
            )
        if not rows:
            return empty("trades")
        df = pd.DataFrame(rows)
        df["ts_utc"] = utc(df["ts_utc"], unit="ms")
        return df

    def _settlements(self, deliveries: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for d in deliveries:
            name = d["instrument_name"]
            if not _NAME.match(name):
                continue
            s = float(d["delivery_price"])
            payoff = self.instrument(name).payoff
            rows.append(
                {
                    "instrument_id": f"deribit:{name}",
                    "settlement_value": float(payoff_value(payoff, s)),
                    "underlying_settlement_price": s,
                    "source": "deribit:delivery",
                }
            )
        return pd.DataFrame(rows) if rows else empty("settlements")
