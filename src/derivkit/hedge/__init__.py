"""Hedges held against target positions. Static (hold to expiry) for now; dynamic delta
hedging with futures or perpetuals is the planned comparison."""

from derivkit.hedge.static import Hedge, Leg, bound_hedge, replicating_hedge, straddling_spread

__all__ = ["Hedge", "Leg", "bound_hedge", "replicating_hedge", "straddling_spread"]
