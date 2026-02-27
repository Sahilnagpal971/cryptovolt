"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any


# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None


class UserResponse(UserBase):
    user_id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Trading Strategy Schemas
class TradingStrategyBase(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class TradingStrategyCreate(TradingStrategyBase):
    pass


class TradingStrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class TradingStrategyResponse(TradingStrategyBase):
    strategy_id: int
    user_id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Market Data Schemas
class MarketDataBase(BaseModel):
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


class MarketDataCreate(MarketDataBase):
    timestamp: datetime
    source: str


class MarketDataResponse(MarketDataBase):
    data_id: int
    timestamp: datetime
    source: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# Sentiment Data Schemas
class SentimentDataBase(BaseModel):
    symbol: str
    source: str
    text: str
    sentiment_score: float


class SentimentDataCreate(SentimentDataBase):
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class SentimentDataResponse(SentimentDataBase):
    sentiment_id: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Model Schemas
class ModelBase(BaseModel):
    name: str
    model_type: str
    version: str


class ModelCreate(ModelBase):
    accuracy: float
    precision: float
    recall: float
    auc: float
    metadata: Optional[Dict[str, Any]] = None


class ModelResponse(ModelCreate):
    model_id: int
    trained_on: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


# Signal Schemas
class SignalBase(BaseModel):
    symbol: str
    signal_type: str
    confidence: float


class SignalCreate(SignalBase):
    strategy_id: int
    model_id: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class SignalResponse(SignalBase):
    signal_id: int
    strategy_id: int
    model_id: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Trade Schemas
class TradeBase(BaseModel):
    symbol: str
    trade_type: str
    price: float
    quantity: float


class TradeCreate(TradeBase):
    signal_id: int
    is_paper_trade: bool = True


class TradeResponse(TradeBase):
    trade_id: int
    signal_id: int
    status: str
    is_paper_trade: bool
    pnl: Optional[float] = None
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


# Alert Schemas
class AlertBase(BaseModel):
    alert_type: str
    message: str


class AlertCreate(AlertBase):
    user_id: int


class AlertResponse(AlertBase):
    alert_id: int
    user_id: int
    is_read: bool
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True
