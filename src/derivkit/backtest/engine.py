"""Event-driven backtest of the two-tier strategy with static hedges held to expiry.

At each decision time (each hedge-chain snapshot) and for each live target contract whose
expiry matches a listed option expiry:

1. build the market state (parity forward and discount factor, SVI smile);
2. compute the fair value, the replication bounds and the target's executable quote;
3. ask the entry rule for a signal; if there is one, open the position with its hedge:
   the bounding spread for tier-1 trades, the strikes either side for tier-2 trades;
4. at settlement, compute P&L and split it exactly into

   edge         q (V - P)                      price vs fair value at entry
   target_fee   -fee                           target venue fee
   hedge_spread mid(H) - exec(H)               paid crossing the hedge's bid/ask
   hedge_fees   -commissions                   option commissions
   pin          q X_T + H(S_T) - (q V + mid H) hedged book's payoff vs its entry value
   basis        H(S_H) - H(S_T)                hedge settled on a different price

where q = +1 (buy) or -1 (sell), X_T is the target's settlement value, H is the hedge book,
S_T the target's settlement price and S_H the hedge's. The six parts sum to the P&L.
Cash flows are not discounted (holding periods are days).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from derivkit.adapters import REGISTRY
from derivkit.config import PairConfig
from derivkit.core.payoffs import Digital, Payoff, payoff_from_record
from derivkit.costs.fees import KalshiFee, fee_in_force
from derivkit.hedge.static import Hedge, bound_hedge, replicating_hedge
from derivkit.market import SECONDS_PER_YEAR, MarketState, MarketStateError, build_state
from derivkit.replicate.bounds import payoff_bounds
from derivkit.replicate.chain import OptionChain
from derivkit.strategy.two_tier import Action, Thresholds, evaluate

TIERS = {"bounds": 1, "model": 2}


@dataclass(frozen=True)
class Settings:
    target_venue: str
    target_series: tuple[str, ...]
    hedge_venue: str
    expiry_tolerance_s: float
    quote_skew_s: float
    tick: float
    thresholds: Thresholds
    allowed_tiers: frozenset[int]
    fee_schedule: tuple[KalshiFee, ...]
    hedge_fee_per_option: float
    order_contracts: int

    @classmethod
    def from_config(cls, cfg: PairConfig, basis_buffer: float | None = None) -> Settings:
        if basis_buffer is None:
            if cfg.strategy.basis_buffer == "measured":
                raise ValueError(
                    f"{cfg.pair}: basis_buffer is 'measured'; estimate the settlement basis "
                    "first and pass it in (CLI: --basis-buffer)"
                )
            basis_buffer = float(cfg.strategy.basis_buffer)
        return cls(
            target_venue=REGISTRY[cfg.target.adapter].venue,
            target_series=tuple(cfg.target.series),
            hedge_venue=REGISTRY[cfg.hedge.adapter].venue,
            expiry_tolerance_s=60.0
            if cfg.match.expiry == "same_close"
            else cfg.match.max_time_skew_seconds,
            quote_skew_s=cfg.match.max_time_skew_seconds,
            tick=cfg.pricing.underlying_tick,
            thresholds=Thresholds(basis_buffer=basis_buffer, min_edge=cfg.strategy.min_edge),
            allowed_tiers=frozenset(TIERS[t] for t in cfg.strategy.tiers),
            fee_schedule=(KalshiFee(date(2000, 1, 1), multiplier=cfg.costs.target_fee_multiplier),),
            hedge_fee_per_option=cfg.costs.hedge_fee_per_option,
            order_contracts=cfg.costs.order_contracts,
        )


@dataclass
class Result:
    trades: pd.DataFrame
    skipped: Counter[str] = field(default_factory=Counter)

    @property
    def settled(self) -> pd.DataFrame:
        return self.trades[self.trades["settled"]] if not self.trades.empty else self.trades


ATTRIBUTION = ["edge", "target_fee", "hedge_spread", "hedge_fees", "pin", "basis"]


def _series_ok(instrument_id: str, venue: str, series: tuple[str, ...]) -> bool:
    if not instrument_id.startswith(f"{venue}:"):
        return False
    if not series:
        return True
    body = instrument_id.split(":", 1)[1]
    return any(body.startswith(f"{s}-") for s in series)


def _chains(quotes: pd.DataFrame, hedge_inst: pd.DataFrame) -> dict[tuple, OptionChain]:
    """(decision ts, expiry, underlying) -> chain, from hedge quotes joined to instruments."""
    cols = ["instrument_id", "strike", "payoff_type", "expiry_utc", "multiplier", "underlying_id"]
    joined = quotes.merge(hedge_inst[cols], on="instrument_id", how="inner")
    out: dict[tuple, OptionChain] = {}
    for key, grp in joined.groupby(["ts_utc", "expiry_utc", "underlying_id"], sort=True):
        grp = grp.drop_duplicates(subset=["strike", "payoff_type"], keep="last")
        try:
            out[key] = OptionChain.from_frame(grp)
        except ValueError:
            continue
    return out


def _latest_quote(
    quotes_by_id: dict[str, pd.DataFrame], instrument_id: str, ts: pd.Timestamp, skew_s: float
) -> tuple[float | None, float | None] | None:
    q = quotes_by_id.get(instrument_id)
    if q is None:
        return None
    pos = q["ts_utc"].searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    row = q.iloc[pos]
    if (ts - row["ts_utc"]).total_seconds() > skew_s:
        return None
    bid, ask = row["bid"], row["ask"]
    return (None if pd.isna(bid) else float(bid), None if pd.isna(ask) else float(ask))


def _hedge_for(
    tier: int, action: Action, payoff: Payoff, state: MarketState, bounds, tick: float
) -> Hedge:
    if tier == 1 and isinstance(payoff, Digital) and payoff.above:
        h = bound_hedge(bounds, "lower" if action is Action.BUY else "upper")
        if h is not None:
            return h
    return replicating_hedge(state.chain, payoff, tick)


def run(
    instruments: pd.DataFrame,
    quotes: pd.DataFrame,
    settlements: pd.DataFrame,
    settings: Settings,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> Result:
    """Run the backtest over decision times in [start, end)."""
    skipped: Counter[str] = Counter()
    targets = instruments[
        [
            _series_ok(i, settings.target_venue, settings.target_series)
            for i in instruments["instrument_id"]
        ]
    ]
    hedge_inst = instruments[
        (instruments["venue"] == settings.hedge_venue)
        & instruments["payoff_type"].isin(["call", "put"])
    ]
    hedge_quotes = quotes[quotes["instrument_id"].isin(hedge_inst["instrument_id"])]
    target_quotes = quotes[quotes["instrument_id"].isin(targets["instrument_id"])].sort_values(
        "ts_utc"
    )
    quotes_by_id = {
        str(k): g.reset_index(drop=True) for k, g in target_quotes.groupby("instrument_id")
    }

    chains = _chains(hedge_quotes, hedge_inst)
    settle = settlements.set_index("instrument_id")
    hedge_settle_price = _hedge_settlement_prices(settlements, hedge_inst)

    rows: list[dict] = []
    held: set[str] = set()
    for (ts, h_expiry, underlying), chain in chains.items():
        if (start is not None and ts < start) or (end is not None and ts >= end):
            continue
        live = targets[
            (targets["underlying_id"] == underlying)
            & (
                (targets["expiry_utc"] - h_expiry).abs().dt.total_seconds()
                <= settings.expiry_tolerance_s
            )
            & (targets["expiry_utc"] > ts)
        ]
        if live.empty:
            continue
        t_years = (h_expiry - ts).total_seconds() / SECONDS_PER_YEAR
        try:
            state = build_state(chain, t_years)
        except (MarketStateError, ValueError) as exc:
            skipped[f"market_state: {type(exc).__name__}"] += len(live)
            continue

        for tgt in live.to_dict("records"):
            iid = tgt["instrument_id"]
            if iid in held:
                skipped["already_held"] += 1
                continue
            quote = _latest_quote(quotes_by_id, iid, ts, settings.quote_skew_s)
            if quote is None:
                skipped["no_target_quote"] += 1
                continue
            payoff = payoff_from_record(tgt["payoff_type"], tgt["strike"], tgt["strike_upper"])
            try:
                fair = state.fair_value(payoff)
                bounds = payoff_bounds(state.chain, payoff, settings.tick)
                replica = replicating_hedge(state.chain, payoff, settings.tick)
            except (KeyError, TypeError):
                skipped["no_hedge_strikes"] += 1
                continue

            fee_model = fee_in_force(settings.fee_schedule, ts.date())
            n = settings.order_contracts

            def fee_per_contract(p: float, _m: KalshiFee = fee_model, _n: int = n) -> float:
                return _m.fee(_n, p) / _n

            hedge_mult = float(hedge_inst["multiplier"].iloc[0])
            commission = settings.hedge_fee_per_option / hedge_mult * replica.option_units()
            spread_cost = _half_spread_cost(replica, state.chain)
            thresholds = Thresholds(
                basis_buffer=settings.thresholds.basis_buffer,
                hedge_fees=commission,
                hedge_spread_cost=spread_cost,
                min_edge=settings.thresholds.min_edge,
            )
            signal = evaluate(
                quote[0],
                quote[1],
                fair,
                bounds,
                fee_per_contract,
                thresholds,
                settings.allowed_tiers,
            )
            if signal.action is Action.NONE or signal.price is None:
                skipped["no_signal"] += 1
                continue

            q = 1.0 if signal.action is Action.BUY else -1.0
            replica_used = _hedge_for(
                signal.tier, signal.action, payoff, state, bounds, settings.tick
            )
            hedge = replica_used.scaled(-q)
            exec_h, mid_h = hedge.prices(state.chain)
            calls = [abs(leg.qty) for leg in hedge.legs if leg.kind == "call"]
            row = {
                "trade_id": f"{iid}@{ts.isoformat()}",
                "decision_ts": ts,
                "instrument_id": iid,
                "expiry_utc": tgt["expiry_utc"],
                "action": signal.action.value,
                "tier": signal.tier,
                "qty": q,  # one target contract paying 1
                "target_price": signal.price,
                "target_fee_paid": fee_per_contract(signal.price),
                "fair_value": fair,
                "lower": bounds.lower.value,
                "upper": bounds.upper.value,
                "expected_edge": signal.edge,
                "hedge": hedge.describe(),
                "hedge_exec": exec_h,
                "hedge_mid": mid_h,
                "hedge_commission": settings.hedge_fee_per_option
                / hedge_mult
                * hedge.option_units(),
                "hedge_block_contracts": hedge_mult / max(calls) if calls else np.nan,
                "forward": state.forward,
                "df": state.df,
                "t_years": t_years,
            }
            row.update(_settle(row, hedge, settle, hedge_settle_price.get(h_expiry)))
            rows.append(row)
            held.add(iid)

    trades = pd.DataFrame(rows)
    return Result(trades=trades, skipped=skipped)


def _half_spread_cost(replica: Hedge, chain: OptionChain) -> float:
    """Average cost of crossing the spread to buy or to sell the replicating portfolio."""
    exec_buy, mid_buy = replica.prices(chain)
    exec_sell, mid_sell = replica.scaled(-1.0).prices(chain)
    return 0.5 * ((exec_buy - mid_buy) + (exec_sell - mid_sell))


def _hedge_settlement_prices(settlements: pd.DataFrame, hedge_inst: pd.DataFrame) -> dict:
    """expiry -> settlement price of the hedge underlying, where the hedge venue reports one."""
    if settlements.empty:
        return {}
    joined = settlements.merge(hedge_inst[["instrument_id", "expiry_utc"]], on="instrument_id")
    joined = joined.dropna(subset=["underlying_settlement_price"])
    return joined.groupby("expiry_utc")["underlying_settlement_price"].first().to_dict()


def _settle(row: dict, hedge: Hedge, settle: pd.DataFrame, s_hedge: float | None) -> dict:
    iid = row["instrument_id"]
    out = {k: np.nan for k in [*ATTRIBUTION, "pnl", "target_payoff", "s_target", "s_hedge"]}
    out.update(settled=False, basis_measured=False)
    if iid not in settle.index:
        return out
    rec = settle.loc[iid]
    x_t = float(rec["settlement_value"])
    s_t = rec["underlying_settlement_price"]
    s_t = None if pd.isna(s_t) else float(s_t)
    if s_t is None and s_hedge is None:
        return out
    basis_measured = s_t is not None and s_hedge is not None
    s_t = s_t if s_t is not None else s_hedge
    s_h = s_hedge if s_hedge is not None else s_t
    assert s_t is not None and s_h is not None

    q, p, v = row["qty"], row["target_price"], row["fair_value"]
    fee = row["target_fee_paid"] * abs(q)
    parts = {
        "edge": q * (v - p),
        "target_fee": -fee,
        "hedge_spread": row["hedge_mid"] - row["hedge_exec"],
        "hedge_fees": -row["hedge_commission"],
        "pin": q * x_t + hedge.payoff(s_t) - (q * v + row["hedge_mid"]),
        "basis": hedge.payoff(s_h) - hedge.payoff(s_t),
    }
    pnl = q * (x_t - p) - fee + hedge.payoff(s_h) - row["hedge_exec"] - row["hedge_commission"]
    out.update(parts)
    out.update(
        pnl=pnl,
        target_payoff=x_t,
        s_target=s_t,
        s_hedge=s_h,
        settled=True,
        basis_measured=basis_measured,
    )
    return out
