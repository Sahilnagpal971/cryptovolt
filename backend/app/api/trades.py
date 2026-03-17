"""Trade execution routes"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import TradeCreate, TradeResponse
from app.models.database import Trade, Signal
from fastapi import HTTPException

router = APIRouter()


@router.post("/", response_model=TradeResponse)
async def execute_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    """Execute new trade"""
    signal = db.query(Signal).filter(Signal.signal_id == trade.signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    status_value = "EXECUTED" if trade.is_paper_trade else "PENDING"
    now = datetime.utcnow()

    db_trade = Trade(
        signal_id=trade.signal_id,
        symbol=trade.symbol.upper().strip(),
        trade_type=trade.trade_type,
        price=trade.price,
        quantity=trade.quantity,
        status=status_value,
        is_paper_trade=trade.is_paper_trade,
        pnl=None,
        timestamp=now,
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)

    return {
        "trade_id": db_trade.trade_id,
        "signal_id": db_trade.signal_id,
        "symbol": db_trade.symbol,
        "trade_type": db_trade.trade_type,
        "price": db_trade.price,
        "quantity": db_trade.quantity,
        "status": db_trade.status,
        "is_paper_trade": db_trade.is_paper_trade,
        "pnl": db_trade.pnl,
        "timestamp": db_trade.timestamp,
        "created_at": db_trade.created_at,
    }


@router.get("/{symbol}")
async def get_trades(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get trades for symbol"""
    symbol = symbol.upper().strip()
    rows = (
        db.query(Trade)
        .filter(Trade.symbol == symbol)
        .order_by(Trade.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "symbol": symbol,
        "trades": [
            {
                "trade_id": r.trade_id,
                "signal_id": r.signal_id,
                "symbol": r.symbol,
                "trade_type": r.trade_type,
                "price": r.price,
                "quantity": r.quantity,
                "status": r.status,
                "is_paper_trade": r.is_paper_trade,
                "pnl": r.pnl,
                "timestamp": r.timestamp,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }
