"""Command line: ``derivkit backtest CONFIG --data DIR [--out DIR] [--open-holdout]``.

``--data`` is a folder of standard-table CSVs (instruments, quotes, settlements) holding
both legs of the pair. By default only the development period runs; the final holdout
(the last ``splits.holdout_fraction`` of decision dates) runs only with ``--open-holdout``,
which should happen once, at the end of the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from derivkit.adapters.csv import CsvAdapter
from derivkit.backtest.engine import Settings, run
from derivkit.config import load_pair
from derivkit.evaluate.metrics import summarise
from derivkit.evaluate.splits import walk_forward


def _decision_dates(quotes: pd.DataFrame) -> list:
    ts = pd.to_datetime(quotes["ts_utc"], utc=True).dt.tz_convert("America/New_York")
    return sorted(set(ts.dt.date))


def backtest(args: argparse.Namespace) -> int:
    cfg = load_pair(args.config)
    tables = CsvAdapter().normalize({"directory": args.data})
    settings = Settings.from_config(cfg, basis_buffer=args.basis_buffer)

    plan = walk_forward(
        _decision_dates(tables.quotes), holdout_fraction=cfg.splits.holdout_fraction
    )
    cutoff = None
    if plan.holdout_start is not None:
        cutoff = pd.Timestamp(plan.holdout_start, tz="America/New_York").tz_convert("UTC")
    start, end = (cutoff, None) if args.open_holdout else (None, cutoff)
    period = "HOLDOUT" if args.open_holdout else "development"
    if args.open_holdout:
        print("Opening the holdout. Do this once, after all choices are frozen.", file=sys.stderr)

    result = run(tables.instruments, tables.quotes, tables.settlements, settings, start, end)
    summary = summarise(result.trades) if not result.trades.empty else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out / f"trades_{period.lower()}.csv", index=False)
    report = {
        "pair": cfg.pair,
        "period": period,
        "holdout_start": str(plan.holdout_start),
        "summary": summary.as_dict() if summary else None,
        "skipped": dict(result.skipped),
    }
    (out / f"summary_{period.lower()}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="derivkit")
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest", help="run a pair backtest on standard-table CSVs")
    bt.add_argument("config", help="pair config YAML")
    bt.add_argument("--data", required=True, help="folder of standard-table CSVs")
    bt.add_argument("--out", default="results", help="output folder")
    bt.add_argument("--basis-buffer", type=float, default=None, help="override basis buffer")
    bt.add_argument("--open-holdout", action="store_true", help="run on the locked holdout")
    bt.set_defaults(func=backtest)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
