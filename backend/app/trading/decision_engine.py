"""
Trading Decision Engine - Core trading logic
"""
import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DecisionEngine:
    """
    Hybrid decision engine that fuses:
    - Technical indicators (rule-based)
    - ML model predictions
    - Sentiment analysis
    - Risk management rules
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the decision engine"""
        self.config = config or {}
        self.ml_weight = self.config.get("ml_weight", 0.6)
        self.rule_weight = self.config.get("rule_weight", 0.3)
        self.sentiment_weight = self.config.get("sentiment_weight", 0.1)
        self.risk_veto_threshold = self.config.get("risk_veto_threshold", 0.7)
        
    def make_decision(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        rule_signals: Dict[str, Any],
        sentiment_score: float,
        volatility: float,
    ) -> Dict[str, Any]:
        """
        Make trading decision by fusing multiple signals
        
        Args:
            symbol: Trading pair symbol
            market_data: Current market data
            ml_prediction: ML model prediction {'signal': str, 'confidence': float}
            rule_signals: Rule-based signals {'signal': str, 'strength': float}
            sentiment_score: Sentiment analysis score (-1 to 1)
            volatility: Current market volatility
            
        Returns:
            Decision dict with signal, confidence, and reasoning
        """
        
        decision = {
            "symbol": symbol,
            "timestamp": datetime.utcnow(),
            "sources": {
                "ml": ml_prediction,
                "rules": rule_signals,
                "sentiment": sentiment_score,
                "volatility": volatility,
            },
            "reasoning": []
        }
        
        # Calculate weighted score
        ml_score = self._signal_to_score(ml_prediction.get("signal"))
        rule_score = self._signal_to_score(rule_signals.get("signal"))
        sentiment_factor = (sentiment_score + 1) / 2  # Normalize to 0-1
        
        # Weighted fusion
        fused_score = (
            (ml_score * self.ml_weight) +
            (rule_score * self.rule_weight) +
            (sentiment_factor * self.sentiment_weight)
        )
        
        decision["fused_score"] = fused_score
        
        # Risk veto check
        if self._should_veto_for_risk(volatility, sentiment_score):
            decision["signal"] = SignalType.HOLD
            decision["confidence"] = 0.5
            decision["reasoning"].append("Risk veto triggered: High volatility + negative sentiment")
            logger.warning(f"Risk veto for {symbol}: volatility={volatility}, sentiment={sentiment_score}")
            return decision
        
        # Generate final signal
        if fused_score >= 0.6:
            decision["signal"] = SignalType.BUY
            decision["confidence"] = min(fused_score, 1.0)
        elif fused_score <= 0.4:
            decision["signal"] = SignalType.SELL
            decision["confidence"] = 1 - fused_score
        else:
            decision["signal"] = SignalType.HOLD
            decision["confidence"] = 0.5
        
        # Add reasoning
        decision["reasoning"].append(f"ML weight: {ml_score:.2f} ({ml_prediction.get('signal')})")
        decision["reasoning"].append(f"Rule weight: {rule_score:.2f} ({rule_signals.get('signal')})")
        decision["reasoning"].append(f"Sentiment: {sentiment_score:.2f}")
        decision["reasoning"].append(f"Fused score: {fused_score:.2f}")
        
        return decision
    
    def _signal_to_score(self, signal: str) -> float:
        """Convert signal to numerical score"""
        signal_map = {
            SignalType.BUY: 1.0,
            SignalType.HOLD: 0.5,
            SignalType.SELL: 0.0,
        }
        return signal_map.get(signal, 0.5)
    
    def _should_veto_for_risk(self, volatility: float, sentiment_score: float) -> bool:
        """Check if risk conditions require veto"""
        # Veto if both high volatility AND very negative sentiment
        high_volatility = volatility > self.config.get("volatility_threshold", 0.05)
        negative_sentiment = sentiment_score < self.config.get("sentiment_threshold", -0.5)
        
        return high_volatility and negative_sentiment


class RiskManager:
    """
    Risk management: position limits, stop-loss, exposure controls
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize risk manager"""
        self.config = config or {}
        self.max_position_size = self.config.get("max_position_size", 1000)
        self.max_daily_loss = self.config.get("max_daily_loss", 500)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 2.0)
        
    def validate_trade(
        self,
        symbol: str,
        trade_type: str,
        quantity: float,
        current_positions: Dict[str, float],
        daily_pnl: float,
    ) -> Dict[str, Any]:
        """
        Validate if trade meets risk criteria
        
        Returns:
            {'approved': bool, 'reason': str, 'adjusted_quantity': float}
        """
        
        result = {
            "approved": True,
            "reason": "Trade approved",
            "adjusted_quantity": quantity,
        }
        
        # Check position size limit
        if quantity > self.max_position_size:
            result["approved"] = False
            result["reason"] = f"Exceeds max position size: {self.max_position_size}"
            return result
        
        # Check daily loss limit
        if daily_pnl <= -self.max_daily_loss:
            result["approved"] = False
            result["reason"] = f"Daily loss limit breached: {daily_pnl:.2f}"
            return result
        
        # Check existing position
        current_position = current_positions.get(symbol, 0)
        if trade_type == "SELL" and current_position <= 0:
            result["approved"] = False
            result["reason"] = f"No position to sell: current={current_position}"
        
        return result
