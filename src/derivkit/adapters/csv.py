"""Adapter for already-normalised tables stored as CSV files (fixtures, exports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from derivkit.adapters.base import Adapter, Tables
from derivkit.core.schemas import TABLES


class CsvAdapter(Adapter):
    venue = "csv"

    def normalize(self, raw: dict[str, Any]) -> Tables:
        """``raw["directory"]``: folder holding ``<table>.csv`` files; missing files stay empty."""
        folder = Path(raw["directory"])
        tables = Tables()
        for name, spec in TABLES.items():
            path = folder / f"{name}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for col in spec.columns:
                if col.kind == "datetime":
                    df[col.name] = pd.to_datetime(df[col.name], utc=True)
                elif col.kind == "bool":
                    df[col.name] = df[col.name].astype(bool)
                elif col.kind == "float":
                    df[col.name] = df[col.name].astype(float)
            setattr(tables, name, df)
        return tables.validated()
