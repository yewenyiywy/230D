"""derivkit: replication-based derivatives pricing and trading.

Layers (each reads only the one before it):

    adapters -> core tables -> calibrate -> price -> replicate -> strategy -> backtest -> evaluate

Only ``adapters`` and the pair config files know which venue or asset class is in use.
"""

__version__ = "0.1.0"
