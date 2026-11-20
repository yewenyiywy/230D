"""Venue adapters. Each one fetches raw data and normalises it to the standard tables."""

from derivkit.adapters.base import Adapter, Tables
from derivkit.adapters.csv import CsvAdapter
from derivkit.adapters.deribit import DeribitAdapter
from derivkit.adapters.kalshi import KalshiAdapter
from derivkit.adapters.optionmetrics import OptionMetricsAdapter

REGISTRY: dict[str, type[Adapter]] = {
    "kalshi": KalshiAdapter,
    "deribit": DeribitAdapter,
    "optionmetrics": OptionMetricsAdapter,
    "csv": CsvAdapter,
}

__all__ = [
    "REGISTRY",
    "Adapter",
    "CsvAdapter",
    "DeribitAdapter",
    "KalshiAdapter",
    "OptionMetricsAdapter",
    "Tables",
]
