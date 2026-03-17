"""Trading strategies routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import TradingStrategyCreate, TradingStrategyResponse
from app.models.database import TradingStrategy, User

router = APIRouter()


@router.post("/", response_model=TradingStrategyResponse)
async def create_strategy(strategy: TradingStrategyCreate, user_id: int = 1, db: Session = Depends(get_db)):
    """Create new trading strategy"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db_strategy = TradingStrategy(
        user_id=user_id,
        name=strategy.name,
        description=strategy.description,
        parameters=strategy.parameters,
        is_active=False,
    )
    db.add(db_strategy)
    db.commit()
    db.refresh(db_strategy)
    return db_strategy


@router.get("/{strategy_id}", response_model=TradingStrategyResponse)
async def get_strategy(strategy_id: int, db: Session = Depends(get_db)):
    """Get strategy by ID"""
    strategy = db.query(TradingStrategy).filter(TradingStrategy.strategy_id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy
