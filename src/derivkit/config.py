"""Pair configuration: one YAML file per target-hedge pair (see ``configs/pairs``).

Moving to a new venue or asset class means a new adapter plus one of these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Leg(_Strict):
    adapter: str
    series: list[str] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list)


class Match(_Strict):
    expiry: Literal["same_close", "same_timestamp"]
    max_time_skew_seconds: float = 0.0
    settlement_hour_utc: int | None = None


class Pricing(_Strict):
    calibrator: Literal["svi", "spline"] = "svi"
    digital_engine: Literal["smile_consistent", "flat_black"] = "smile_consistent"
    underlying_tick: float = 0.0


class Strategy(_Strict):
    tiers: list[Literal["bounds", "model"]] = Field(default_factory=lambda: ["bounds", "model"])
    hold: Literal["to_expiry", "until_converged"] = "to_expiry"
    basis_buffer: float | Literal["measured"] = 0.0
    min_edge: float = 0.0


class Costs(_Strict):
    fees: Literal["by_date"] = "by_date"
    fill: Literal["cross_spread"] = "cross_spread"
    depth_cap: bool = True
    target_fee_multiplier: float = 1.0  # per the venue's schedule for this series
    hedge_fee_per_option: float = 0.0  # commission per listed option contract, USD
    order_contracts: int = Field(default=100, gt=0)  # order size used for fee rounding


class Splits(_Strict):
    scheme: Literal["walk_forward"] = "walk_forward"
    holdout_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)


class PairConfig(_Strict):
    pair: str
    target: Leg
    hedge: Leg
    match: Match
    pricing: Pricing = Field(default_factory=Pricing)
    strategy: Strategy = Field(default_factory=Strategy)
    costs: Costs = Field(default_factory=Costs)
    splits: Splits = Field(default_factory=lambda: Splits())


def load_pair(path: str | Path) -> PairConfig:
    with open(path) as fh:
        return PairConfig.model_validate(yaml.safe_load(fh))
