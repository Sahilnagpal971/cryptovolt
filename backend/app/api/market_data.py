"""Market data routes"""
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import MarketDataResponse
from app.core.config import settings
from app.data.ingestion import BinanceDataIngestion
from app.models.database import MarketData

router = APIRouter()


@router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    limit: int = 100,
    interval: str = "1m",
    source: Literal["db", "binance"] = "db",
    db: Session = Depends(get_db),
):
    """Get candlestick data for symbol"""
    symbol = symbol.upper().strip()

    if source == "db":
        rows = (
            db.query(MarketData)
            .filter(MarketData.symbol == symbol)
            .order_by(MarketData.timestamp.desc())
            .limit(limit)
            .all()
        )
        candles = [
            {
                "open_time": r.timestamp,
                "open": r.open_price,
                "high": r.high_price,
                "low": r.low_price,
                "close": r.close_price,
                "volume": r.volume,
                "source": r.source,
            }
            for r in reversed(rows)
        ]
        return {"symbol": symbol, "interval": interval, "source": "db", "candles": candles}

    ingestion = BinanceDataIngestion(
        api_key=settings.BINANCE_API_KEY,
        api_secret=settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )
    try:
        candles = await ingestion.fetch_historical_candles(symbol=symbol, interval=interval, limit=limit)
        return {"symbol": symbol, "interval": interval, "source": "binance", "candles": candles}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            await ingestion.close()
        except Exception:
            pass


@router.get("/data/{symbol}", response_model=list[MarketDataResponse])
async def get_market_data(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get market data for symbol"""
    symbol = symbol.upper().strip()
    return (
        db.query(MarketData)
        .filter(MarketData.symbol == symbol)
        .order_by(MarketData.timestamp.desc())
        .limit(limit)
        .all()
    )
