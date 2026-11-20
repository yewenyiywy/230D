"""One expiry's executable option quotes, the input to every replication routine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

F64 = NDArray[np.floating]


@dataclass(frozen=True)
class OptionChain:
    """Calls (and optionally puts) for a single expiry, sorted by strike.

    Prices are present values in USD per unit of underlying (divide by the multiplier
    first). Missing quotes are NaN.
    """

    strikes: F64
    call_bid: F64
    call_ask: F64
    put_bid: F64 | None = None
    put_ask: F64 | None = None
    df: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.strikes)
        for name in ("call_bid", "call_ask"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length differs from strikes")
        if np.any(np.diff(self.strikes) <= 0):
            raise ValueError("strikes must be strictly increasing")

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        discount: float = 1.0,
        strike: str = "strike",
        option_type: str = "payoff_type",
        bid: str = "bid",
        ask: str = "ask",
        multiplier: str | None = "multiplier",
    ) -> OptionChain:
        """Build from a joined quotes + instruments frame for one expiry and timestamp."""
        frame = df.copy()
        if multiplier and multiplier in frame:
            frame[bid] = frame[bid] / frame[multiplier]
            frame[ask] = frame[ask] / frame[multiplier]
        calls = frame[frame[option_type] == "call"].set_index(strike)
        puts = frame[frame[option_type] == "put"].set_index(strike)
        ks = np.array(sorted(set(calls.index) | set(puts.index)), dtype=float)

        def col(side: pd.DataFrame, name: str) -> F64:
            return side[name].reindex(ks).to_numpy(dtype=float)

        has_puts = not puts.empty
        return cls(
            strikes=ks,
            call_bid=col(calls, bid),
            call_ask=col(calls, ask),
            put_bid=col(puts, bid) if has_puts else None,
            put_ask=col(puts, ask) if has_puts else None,
            df=discount,
        )
