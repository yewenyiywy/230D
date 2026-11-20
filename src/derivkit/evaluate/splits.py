"""Time-ordered splits: expanding walk-forward folds plus a locked final holdout.

The holdout is the last ``holdout_fraction`` of decision dates. Nothing in the
development folds may touch it; the CLI only runs on it when asked explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: tuple[date, ...]
    test: tuple[date, ...]


@dataclass(frozen=True)
class Plan:
    development: tuple[date, ...]
    holdout: tuple[date, ...]
    folds: tuple[Fold, ...]

    @property
    def holdout_start(self) -> date | None:
        return self.holdout[0] if self.holdout else None


def walk_forward(dates: Sequence[date], n_folds: int = 4, holdout_fraction: float = 0.2) -> Plan:
    days = sorted(set(dates))
    if not days:
        return Plan((), (), ())
    n_hold = int(np.ceil(len(days) * holdout_fraction))
    dev, hold = days[: len(days) - n_hold], days[len(days) - n_hold :]
    chunks = [list(c) for c in np.array_split(np.array(dev, dtype=object), n_folds + 1) if len(c)]
    folds = tuple(
        Fold(tuple(d for c in chunks[: i + 1] for d in c), tuple(chunks[i + 1]))
        for i in range(len(chunks) - 1)
    )
    return Plan(tuple(dev), tuple(hold), folds)
