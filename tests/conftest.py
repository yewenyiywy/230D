from __future__ import annotations

import numpy as np
import pytest

from derivkit.calibrate.svi import SVI
from derivkit.price.smile_digital import svi_call
from derivkit.replicate.chain import OptionChain

FORWARD = 7400.0
T = 1.0 / 252.0
DF = float(np.exp(-0.04 * T))


@pytest.fixture
def skewed_svi() -> SVI:
    """A one-day equity-index-like smile: ~14% ATM, ~22% at -3%, ~15% at +3% moneyness."""
    atm_w = 0.15**2 * T
    return SVI(a=0.5 * atm_w, b=0.003, rho=-0.6, m=0.0, s=0.01, t=T)


def make_chain(
    svi: SVI, step: float = 5.0, half_spread: float = 0.05, width: int = 40
) -> OptionChain:
    strikes = FORWARD + step * np.arange(-width, width + 1)
    mid = svi_call(strikes, FORWARD, svi, DF)
    puts = mid - DF * (FORWARD - strikes)  # put-call parity
    return OptionChain(
        strikes=strikes,
        call_bid=np.maximum(mid - half_spread, 0.0),
        call_ask=mid + half_spread,
        put_bid=np.maximum(puts - half_spread, 0.0),
        put_ask=puts + half_spread,
        df=DF,
    )


@pytest.fixture
def chain(skewed_svi: SVI) -> OptionChain:
    return make_chain(skewed_svi)
