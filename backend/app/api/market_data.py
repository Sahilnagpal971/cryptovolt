"""Market data routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import MarketDataResponse

router = APIRouter()


@router.get("/candles/{symbol}")
async def get_candles(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get candlestick data for symbol"""
    # TODO: Implement candle retrieval
    return {"symbol": symbol, "candles": []}


@router.get("/data/{symbol}", response_model=list[MarketDataResponse])
async def get_market_data(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get market data for symbol"""
    # TODO: Implement market data retrieval
    return []
