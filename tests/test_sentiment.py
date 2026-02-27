"""Unit tests for sentiment analyzer"""
import pytest
from app.sentiment.analyzer import EnhancedCryptoSentimentAnalyzer


@pytest.fixture
def sentiment_analyzer():
    """Create sentiment analyzer instance"""
    return EnhancedCryptoSentimentAnalyzer()


def test_analyze_sentiment_positive(sentiment_analyzer):
    """Test positive sentiment analysis"""
    text = "Bitcoin surges to new highs! Great investment opportunity. Moon incoming!"
    
    result = sentiment_analyzer.hybrid_sentiment_analysis(text)
    
    assert "compound" in result
    assert -1.0 <= result["compound"] <= 1.0
    assert "confidence" in result
    assert "crypto_adjustment" in result


def test_analyze_sentiment_negative(sentiment_analyzer):
    """Test negative sentiment analysis"""
    text = "Market crash! Bitcoin plummeting. Sell everything now. Major dump happening."
    
    result = sentiment_analyzer.hybrid_sentiment_analysis(text)
    
    assert "compound" in result
    assert -1.0 <= result["compound"] <= 1.0


def test_analyze_sentiment_neutral(sentiment_analyzer):
    """Test neutral sentiment"""
    text = "Bitcoin price remains stable at current levels."
    
    result = sentiment_analyzer.hybrid_sentiment_analysis(text)
    
    assert "compound" in result
    assert -1.0 <= result["compound"] <= 1.0
