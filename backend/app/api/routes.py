"""
API Routes Module
"""
from . import (
    health,
    auth,
    users,
    strategies,
    market_data,
    sentiment,
    signals,
    trades,
    models,
    alerts,
)

__all__ = [
    "health",
    "auth",
    "users",
    "strategies",
    "market_data",
    "sentiment",
    "signals",
    "trades",
    "models",
    "alerts",
]
