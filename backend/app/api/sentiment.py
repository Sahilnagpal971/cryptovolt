"""Sentiment analysis routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import SentimentDataResponse

router = APIRouter()


@router.get("/score/{symbol}")
async def get_sentiment_score(symbol: str, db: Session = Depends(get_db)):
    """Get current sentiment score for symbol"""
    # TODO: Implement sentiment score calculation
    return {"symbol": symbol, "score": 0.5, "timestamp": "2024-01-01"}


@router.get("/data/{symbol}", response_model=list[SentimentDataResponse])
async def get_sentiment_data(symbol: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get sentiment data for symbol"""
    # TODO: Implement sentiment data retrieval
    return []
