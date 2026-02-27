"""Unit tests for sentiment analyzer"""
import pytest
from app.sentiment.analyzer import SentimentAnalyzer


@pytest.fixture
def sentiment_analyzer():
    """Create sentiment analyzer instance"""
    config = {
        "aggregation_method": "weighted_average",
    }
    return SentimentAnalyzer(config)


def test_analyze_sentiment_positive(sentiment_analyzer):
    """Test positive sentiment analysis"""
    sentiment_data = [
        {"source": "NEWS", "sentiment_score": 0.8, "text": "Bitcoin surges", "timestamp": "2024-01-01"},
        {"source": "NEWS", "sentiment_score": 0.7, "text": "Positive outlook", "timestamp": "2024-01-01"},
    ]
    
    result = sentiment_analyzer.analyze_symbol_sentiment("BTCUSDT", sentiment_data)
    
    assert result["overall_score"] > 0
    assert result["trend"] in ["POSITIVE", "VERY_POSITIVE"]
    assert result["sources_analyzed"] == 2


def test_analyze_sentiment_negative(sentiment_analyzer):
    """Test negative sentiment analysis"""
    sentiment_data = [
        {"source": "REDDIT", "sentiment_score": -0.8, "text": "Market crash", "timestamp": "2024-01-01"},
    ]
    
    result = sentiment_analyzer.analyze_symbol_sentiment("BTCUSDT", sentiment_data)
    
    assert result["overall_score"] < 0
    assert result["trend"] in ["NEGATIVE", "VERY_NEGATIVE"]


def test_analyze_sentiment_empty(sentiment_analyzer):
    """Test empty sentiment data"""
    result = sentiment_analyzer.analyze_symbol_sentiment("BTCUSDT", [])
    
    assert result["overall_score"] == 0.0
    assert result["sources_analyzed"] == 0
