"""Sentiment analysis routes"""
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.sentiment.analyzer import EnhancedCryptoSentimentAnalyzer

router = APIRouter()

_analyzer: EnhancedCryptoSentimentAnalyzer | None = None


def _get_analyzer() -> EnhancedCryptoSentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        reddit_config = {
            "client_id": settings.REDDIT_CLIENT_ID,
            "client_secret": settings.REDDIT_CLIENT_SECRET,
            "user_agent": settings.REDDIT_USER_AGENT,
        }
        _analyzer = EnhancedCryptoSentimentAnalyzer(
            reddit_config=reddit_config,
            use_finbert=settings.USE_FINBERT,
            cache_ttl=settings.SENTIMENT_CACHE_TTL,
        )
    return _analyzer


def _normalize_symbol(symbol: str) -> tuple[str, str]:
    symbol = symbol.upper().strip()
    if symbol.endswith("USDT"):
        base = symbol[:-4]
    else:
        base = symbol

    mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binance",
        "SOL": "solana",
        "ADA": "cardano",
    }

    return mapping.get(base, base.lower()), base


@router.get("/score/{symbol}")
async def get_sentiment_score(symbol: str):
    """Get current sentiment score for symbol"""
    try:
        coin_name, coin_id = _normalize_symbol(symbol)
        analyzer = _get_analyzer()
        result = analyzer.get_combined_market_sentiment(coin=coin_name, coin_id=coin_id)
        return {
            "symbol": symbol.upper(),
            "sentiment": result["final_sentiment"],
            "score": result["final_score"],
            "confidence": result["final_confidence"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/combined/{symbol}")
async def get_combined_sentiment(symbol: str, reddit_limit: int = 150, news_limit: int = 150):
    """Get combined sentiment analysis for symbol"""
    try:
        coin_name, coin_id = _normalize_symbol(symbol)
        analyzer = _get_analyzer()
        return analyzer.get_combined_market_sentiment(
            coin=coin_name,
            coin_id=coin_id,
            reddit_limit=reddit_limit,
            news_limit=news_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze")
async def analyze_text(text: str):
    """Analyze sentiment for a single text input"""
    try:
        analyzer = _get_analyzer()
        return analyzer.hybrid_sentiment_analysis(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
