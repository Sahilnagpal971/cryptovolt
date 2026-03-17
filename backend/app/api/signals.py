"""Trading signals routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import SignalCreate, SignalResponse
from app.models.database import Signal, TradingStrategy, Model
from fastapi import HTTPException

router = APIRouter()


@router.post("/", response_model=SignalResponse)
async def create_signal(signal: SignalCreate, db: Session = Depends(get_db)):
    """Create new trading signal"""
    strategy = db.query(TradingStrategy).filter(TradingStrategy.strategy_id == signal.strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    model = db.query(Model).filter(Model.model_id == signal.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db_signal = Signal(
        strategy_id=signal.strategy_id,
        model_id=signal.model_id,
        symbol=signal.symbol.upper().strip(),
        signal_type=signal.signal_type,
        confidence=signal.confidence,
        timestamp=signal.timestamp,
        extra_data=signal.metadata,
    )
    db.add(db_signal)
    db.commit()
    db.refresh(db_signal)

    return {
        "signal_id": db_signal.signal_id,
        "strategy_id": db_signal.strategy_id,
        "model_id": db_signal.model_id,
        "symbol": db_signal.symbol,
        "signal_type": db_signal.signal_type,
        "confidence": db_signal.confidence,
        "timestamp": db_signal.timestamp,
        "metadata": db_signal.extra_data,
        "created_at": db_signal.created_at,
    }


@router.get("/{symbol}")
async def get_signals(symbol: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get signals for symbol"""
    symbol = symbol.upper().strip()
    rows = (
        db.query(Signal)
        .filter(Signal.symbol == symbol)
        .order_by(Signal.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "symbol": symbol,
        "signals": [
            {
                "signal_id": r.signal_id,
                "strategy_id": r.strategy_id,
                "model_id": r.model_id,
                "symbol": r.symbol,
                "signal_type": r.signal_type,
                "confidence": r.confidence,
                "timestamp": r.timestamp,
                "metadata": r.extra_data,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }
