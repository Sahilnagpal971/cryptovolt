"""Unit tests for decision engine"""
import pytest
from app.trading.decision_engine import DecisionEngine, SignalType


@pytest.fixture
def decision_engine():
    """Create decision engine instance"""
    config = {
        "ml_weight": 0.6,
        "rule_weight": 0.3,
        "sentiment_weight": 0.1,
        "volatility_threshold": 0.05,
        "sentiment_threshold": -0.5,
    }
    return DecisionEngine(config)


def test_make_decision_buy(decision_engine):
    """Test buy signal generation"""
    decision = decision_engine.make_decision(
        symbol="BTCUSDT",
        market_data={"open": 45000, "close": 46000},
        ml_prediction={"signal": SignalType.BUY, "confidence": 0.8},
        rule_signals={"signal": SignalType.BUY, "strength": 0.7},
        sentiment_score=0.5,
        volatility=0.02,
    )
    
    assert decision["signal"] == SignalType.BUY
    assert decision["confidence"] >= 0.6


def test_make_decision_sell(decision_engine):
    """Test sell signal generation"""
    decision = decision_engine.make_decision(
        symbol="BTCUSDT",
        market_data={"open": 46000, "close": 45000},
        ml_prediction={"signal": SignalType.SELL, "confidence": 0.8},
        rule_signals={"signal": SignalType.SELL, "strength": 0.7},
        sentiment_score=-0.5,
        volatility=0.02,
    )
    
    assert decision["signal"] == SignalType.SELL


def test_risk_veto(decision_engine):
    """Test risk veto mechanism"""
    decision = decision_engine.make_decision(
        symbol="BTCUSDT",
        market_data={"open": 45000, "close": 46000},
        ml_prediction={"signal": SignalType.BUY, "confidence": 0.9},
        rule_signals={"signal": SignalType.BUY, "strength": 0.9},
        sentiment_score=-0.7,  # Very negative sentiment
        volatility=0.1,        # High volatility
    )
    
    # Should be vetoed despite strong buy signals
    assert decision["signal"] == SignalType.HOLD
