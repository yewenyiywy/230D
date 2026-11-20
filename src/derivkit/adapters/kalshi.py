"""Kalshi Trade API v2 adapter.

Public market data needs no key. Endpoints used:
  GET /markets?series_ticker=...                       contracts and results
  GET /series/{series}/markets/{ticker}/candlesticks   top-of-book bid/ask history
  GET /markets/trades?ticker=...                       trades
  GET /historical/...                                  same data past the live cutoff

Prices: Kalshi has moved from integer cents (``yes_bid``) to dollar strings
(``yes_bid_dollars``, ``close_dollars``); both are accepted.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from derivkit.adapters.base import Adapter, Tables, utc
from derivkit.core.instruments import AssetClass, Instrument, SettlementKind, SettlementRule
from derivkit.core.payoffs import Digital, Payoff, Range
from derivkit.core.schemas import empty

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Series we use, with the settlement rule printed in each contract's terms.
SERIES: dict[str, tuple[str, AssetClass, SettlementRule]] = {
    "KXINX": (
        "SPX",
        AssetClass.INDEX,
        SettlementRule(SettlementKind.CLOSE, "kalshi:sp500_close"),
    ),
    "KXINXU": (
        "SPX",
        AssetClass.INDEX,
        SettlementRule(SettlementKind.CLOSE, "kalshi:sp500_close"),
    ),
    "KXBTCD": (
        "BTC",
        AssetClass.CRYPTO,
        # 60 one-second BRTI prints before the hour, top and bottom 20% dropped.
        SettlementRule(SettlementKind.TRIMMED_MEAN, "cfbenchmarks:brti", 60, 0.2),
    ),
}


def _price(obj: dict[str, Any], field: str) -> float | None:
    """Read a dollar price from either ``<field>_dollars`` (string) or ``<field>`` (cents)."""
    dollars = obj.get(f"{field}_dollars")
    if dollars not in (None, ""):
        return float(dollars)
    cents = obj.get(field)
    if cents is None:
        return None
    return float(cents) / 100.0


def _ohlc_close(obj: dict[str, Any] | None) -> float | None:
    if not obj:
        return None
    if obj.get("close_dollars") not in (None, ""):
        return float(obj["close_dollars"])
    if obj.get("close") is not None:
        return float(obj["close"]) / 100.0
    return None


def series_of(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def market_payoff(market: dict[str, Any]) -> Payoff:
    """Payoff from Kalshi's ``strike_type`` / ``floor_strike`` / ``cap_strike`` fields."""
    strike_type = market.get("strike_type")
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    if strike_type in ("greater", "greater_or_equal") and floor is not None:
        return Digital(float(floor), above=True)
    if strike_type in ("less", "less_or_equal") and cap is not None:
        return Digital(float(cap), above=False)
    if strike_type == "between":
        return Range(
            float(floor) if floor is not None else None,
            float(cap) if cap is not None else None,
        )
    raise ValueError(f"{market.get('ticker')}: unsupported strike_type {strike_type!r}")


class KalshiAdapter(Adapter):
    venue = "kalshi"

    def __init__(self, base_url: str = BASE_URL, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout

    # ---------------------------------------------------------------- fetch (network)
    def _get(self, client: httpx.Client, path: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = client.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _paged(self, path: str, params: dict[str, Any], key: str) -> Iterator[dict[str, Any]]:
        params = dict(params)
        with httpx.Client(timeout=self.timeout) as client:
            while True:
                page = self._get(client, path, params)
                yield from page.get(key, [])
                cursor = page.get("cursor")
                if not cursor:
                    return
                params["cursor"] = cursor

    def fetch_markets(self, series_ticker: str, status: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": 1000}
        if status:
            params["status"] = status
        return list(self._paged("/markets", params, "markets"))

    def fetch_candlesticks(
        self, ticker: str, start: datetime, end: datetime, period_minutes: int = 1
    ) -> list[dict]:
        path = f"/series/{series_of(ticker)}/markets/{ticker}/candlesticks"
        params = {
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        }
        with httpx.Client(timeout=self.timeout) as client:
            return self._get(client, path, params).get("candlesticks", [])

    def fetch_trades(self, ticker: str) -> list[dict]:
        return list(self._paged("/markets/trades", {"ticker": ticker, "limit": 1000}, "trades"))

    # ------------------------------------------------------------- normalize (pure)
    def normalize(self, raw: dict[str, Any]) -> Tables:
        """``raw`` keys: ``markets`` (list), ``candlesticks`` ({ticker: list}), ``trades``."""
        tables = Tables()
        markets = raw.get("markets", [])
        if markets:
            tables.instruments = self.instruments_frame([self.instrument(m) for m in markets])
            tables.settlements = self._settlements(markets)
        if raw.get("candlesticks"):
            tables.quotes = self._quotes(raw["candlesticks"])
        if raw.get("trades"):
            tables.trades = self._trades(raw["trades"])
        return tables.validated()

    def instrument(self, market: dict[str, Any]) -> Instrument:
        ticker = market["ticker"]
        series = series_of(ticker)
        if series not in SERIES:
            raise ValueError(f"{ticker}: series {series} not configured in kalshi.SERIES")
        underlying, asset_class, rule = SERIES[series]
        expiry = pd.Timestamp(market["close_time"]).tz_convert("UTC").to_pydatetime()
        return Instrument(
            instrument_id=f"kalshi:{ticker}",
            venue=self.venue,
            underlying_id=underlying,
            asset_class=asset_class,
            payoff=market_payoff(market),
            expiry_utc=expiry,
            settlement=rule,
            multiplier=1.0,
        )

    def _settlements(self, markets: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for m in markets:
            result = m.get("result")
            if result not in ("yes", "no"):
                continue
            value = m.get("expiration_value")
            rows.append(
                {
                    "instrument_id": f"kalshi:{m['ticker']}",
                    "settlement_value": 1.0 if result == "yes" else 0.0,
                    "underlying_settlement_price": float(value)
                    if value not in (None, "")
                    else None,
                    "source": "kalshi:markets",
                }
            )
        return pd.DataFrame(rows) if rows else empty("settlements")

    def _quotes(self, candles_by_ticker: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
        rows = []
        for ticker, candles in candles_by_ticker.items():
            for c in candles:
                rows.append(
                    {
                        "ts_utc": c["end_period_ts"],
                        "instrument_id": f"kalshi:{ticker}",
                        "bid": _ohlc_close(c.get("yes_bid")),
                        "ask": _ohlc_close(c.get("yes_ask")),
                        "bid_size": None,
                        "ask_size": None,
                        "source": "kalshi:candlesticks",
                        "synchronised": False,
                    }
                )
        if not rows:
            return empty("quotes")
        df = pd.DataFrame(rows)
        df["ts_utc"] = utc(df["ts_utc"], unit="s")
        for col in ("bid", "ask", "bid_size", "ask_size"):
            df[col] = df[col].astype(float)
        return df

    def _trades(self, trades: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for t in trades:
            size = t.get("count_fp", t.get("count"))
            side = t.get("taker_side")
            rows.append(
                {
                    "ts_utc": t["created_time"],
                    "instrument_id": f"kalshi:{t['ticker']}",
                    "price": _price(t, "yes_price"),
                    "size": float(size) if size is not None else float("nan"),
                    # In YES terms: a taker buying YES lifts the offer.
                    "aggressor_side": {"yes": "buy", "no": "sell"}.get(str(side)),
                    "trade_id": str(t["trade_id"]),
                }
            )
        df = pd.DataFrame(rows)
        df["ts_utc"] = utc(df["ts_utc"])
        return df
