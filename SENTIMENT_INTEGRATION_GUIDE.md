# Sentiment Analyzer - Implementation & Integration Guide

## Quick Start

### 1. Copy Sentiment Analyzer Code

The `EnhancedCryptoSentimentAnalyzer` class should be placed at:
```
d:\cryptovolt\backend\app\sentiment\analyzer.py
```

### 2. Create Supporting Files

```
backend/app/sentiment/
├── __init__.py              # Package initialization
├── analyzer.py              # EnhancedCryptoSentimentAnalyzer (provided code)
├── sentiment_models.py       # Pydantic data models
├── sentiment_service.py      # Business logic wrapper
└── utils.py                 # Helper functions
```

### 3. Install Required Dependencies

```bash
cd d:\cryptovolt

# Activate virtual environment
.\venv\Scripts\activate  # On PowerShell with execution policy bypass

# Install sentiment analysis libraries
d:\cryptovolt\venv\Scripts\python.exe -m pip install --default-timeout=1000 \
    vaderSentiment==3.3.2 \
    praw==7.7.0 \
    transformers==4.35.0 \
    torch==2.4.1
```

Update [requirements.txt](requirements.txt):
```txt
# ... existing packages ...

# Sentiment Analysis
vaderSentiment==3.3.2
praw==7.7.0
transformers==4.35.0  # For FinBERT (optional)
lxml==4.9.3           # For RSS parsing
beautifulsoup4==4.12.2 # For HTML parsing
```

### 4. Configure Environment Variables

Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Update with your credentials:
```env
REDDIT_CLIENT_ID=oXvmcPz6Sb2ObD5q9FQ0dw
REDDIT_CLIENT_SECRET=RbNtP24dpX_S2t19fbIVTlz-AeZYYA
REDDIT_USER_AGENT=CryptoVolt/3.0 by u/Ill-Database-3830

DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1429173388814844026/...
```

---

## Implementation Steps

### Step 1: Create Pydantic Models

**File**: `backend/app/sentiment/sentiment_models.py`

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, List

class SentimentLevel(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class SingleSourceAnalysis(BaseModel):
    """Analysis from a single source (Reddit/News)"""
    sentiment: SentimentLevel
    weighted_score: float = Field(..., ge=-1.0, le=1.0)
    unweighted_score: float = Field(..., ge=-1.0, le=1.0)
    reliability: float = Field(..., ge=0.0, le=1.0)
    std_dev: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    sample_size: int
    target_sample: int
    pos_count: int
    neg_count: int
    neu_count: int
    sarcasm_count: int
    finbert_count: Optional[int] = 0
    source_name: str
    disaggregation: Dict[str, float]

class SourceDivergence(BaseModel):
    """Measure of disagreement between sources"""
    score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "LOW", "MEDIUM", "HIGH"

class CombinedMarketSentiment(BaseModel):
    """Final combined sentiment from all sources"""
    final_sentiment: SentimentLevel
    final_score: float = Field(..., ge=-1.0, le=1.0)
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    total_samples: int
    source_divergence: SourceDivergence
    reddit_analysis: SingleSourceAnalysis
    news_analysis: SingleSourceAnalysis
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HybridSentimentResult(BaseModel):
    """Result from hybrid sentiment analysis of single text"""
    compound: float = Field(..., ge=-1.0, le=1.0)
    pos: float = Field(..., ge=0.0, le=1.0)
    neg: float = Field(..., ge=0.0, le=1.0)
    neu: float = Field(..., ge=0.0, le=1.0)
    base_compound: float
    crypto_adjustment: float
    is_sarcastic: bool
    sarcasm_confidence: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    finbert_used: bool
```

### Step 2: Create Sentiment Service

**File**: `backend/app/sentiment/sentiment_service.py`

```python
import logging
from typing import Optional, Dict
from datetime import datetime
from sentiment.analyzer import EnhancedCryptoSentimentAnalyzer
from sentiment.sentiment_models import CombinedMarketSentiment
from core.config import settings

logger = logging.getLogger(__name__)

class SentimentService:
    """Business logic wrapper for sentiment analysis"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        reddit_config = {
            'client_id': settings.REDDIT_CLIENT_ID,
            'client_secret': settings.REDDIT_CLIENT_SECRET,
            'user_agent': settings.REDDIT_USER_AGENT
        }
        
        self.analyzer = EnhancedCryptoSentimentAnalyzer(
            reddit_config=reddit_config,
            use_finbert=settings.USE_FINBERT
        )
        
        self.logger = logger
    
    def analyze_coin(
        self, 
        coin_name: str,
        coin_id: str = None,
        reddit_limit: int = 150,
        news_limit: int = 150
    ) -> CombinedMarketSentiment:
        """
        Analyze market sentiment for a cryptocurrency.
        
        Args:
            coin_name: Name of coin (bitcoin, ethereum)
            coin_id: Exchange ID (BTC, ETH) - auto-derived if None
            reddit_limit: Number of Reddit posts to collect
            news_limit: Number of news articles to collect
            
        Returns:
            CombinedMarketSentiment with all analysis data
        """
        self.logger.info(f"Starting sentiment analysis for {coin_name.upper()}")
        
        try:
            # If coin_id not provided, derive from name
            if not coin_id:
                coin_id_map = {
                    'bitcoin': 'BTC',
                    'ethereum': 'ETH',
                    'binance': 'BNB',
                    'cardano': 'ADA',
                    'solana': 'SOL'
                }
                coin_id = coin_id_map.get(coin_name.lower(), coin_name.upper()[:3])
            
            # Get combined sentiment
            result = self.analyzer.get_combined_market_sentiment(
                coin=coin_name.lower(),
                coin_id=coin_id.upper(),
                reddit_limit=reddit_limit,
                news_limit=news_limit
            )
            
            # Validate and convert to model
            sentiment_model = CombinedMarketSentiment(**result)
            
            self.logger.info(
                f"✅ Sentiment analysis complete: "
                f"{coin_name} -> {sentiment_model.final_sentiment} "
                f"({sentiment_model.final_score:.4f})"
            )
            
            return sentiment_model
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of custom text"""
        return self.analyzer.hybrid_sentiment_analysis(text)
    
    def get_bitcoin_sentiment(self) -> CombinedMarketSentiment:
        """Get Bitcoin sentiment (shortcut)"""
        return self.analyze_coin('bitcoin', 'BTC')
    
    def get_ethereum_sentiment(self) -> CombinedMarketSentiment:
        """Get Ethereum sentiment (shortcut)"""
        return self.analyze_coin('ethereum', 'ETH')

# Global service instance
_sentiment_service: Optional[SentimentService] = None

def get_sentiment_service() -> SentimentService:
    """Get or create sentiment service (singleton)"""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service
```

### Step 3: Create FastAPI Routes

**File**: `backend/app/api/sentiment.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
from sentiment.sentiment_service import get_sentiment_service
from sentiment.sentiment_models import CombinedMarketSentiment

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])
logger = logging.getLogger(__name__)

@router.get("/bitcoin", response_model=CombinedMarketSentiment)
async def get_bitcoin_sentiment():
    """
    Get real-time sentiment analysis for Bitcoin.
    
    Returns:
        CombinedMarketSentiment with Reddit + News analysis
    """
    try:
        service = get_sentiment_service()
        sentiment = service.get_bitcoin_sentiment()
        return sentiment
    except Exception as e:
        logger.error(f"Bitcoin sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ethereum", response_model=CombinedMarketSentiment)
async def get_ethereum_sentiment():
    """Get real-time sentiment analysis for Ethereum."""
    try:
        service = get_sentiment_service()
        sentiment = service.get_ethereum_sentiment()
        return sentiment
    except Exception as e:
        logger.error(f"Ethereum sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{coin}", response_model=CombinedMarketSentiment)
async def get_coin_sentiment(coin: str):
    """
    Get sentiment analysis for any cryptocurrency.
    
    Args:
        coin: Cryptocurrency name (bitcoin, ethereum, cardano, etc.)
    """
    try:
        service = get_sentiment_service()
        sentiment = service.analyze_coin(coin)
        return sentiment
    except Exception as e:
        logger.error(f"Sentiment analysis for {coin} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_text(text: str):
    """
    Analyze sentiment of custom text.
    
    Args:
        text: Text to analyze
    """
    try:
        service = get_sentiment_service()
        result = service.analyze_text(text)
        return result
    except Exception as e:
        logger.error(f"Text sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def sentiment_health():
    """Check sentiment analyzer health"""
    try:
        service = get_sentiment_service()
        # Quick sanity check
        result = service.analyze_text("Bitcoin is good")
        
        return {
            "status": "healthy",
            "analyzer_loaded": True,
            "test_sentiment": result['compound']
        }
    except Exception as e:
        logger.error(f"Sentiment health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
```

### Step 4: Update Main Application

**File**: `backend/app/main.py`

```python
from fastapi import FastAPI
from api import sentiment, signals, trades, health
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="CryptoVolt Trading API", version="3.0.2")

# Register routers
app.include_router(sentiment.router)
app.include_router(signals.router)
app.include_router(trades.router)
app.include_router(health.router)

@app.get("/")
async def root():
    return {"message": "CryptoVolt API", "version": "3.0.2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
```

### Step 5: Update Config

**File**: `backend/app/core/config.py`

```python
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application configuration from environment variables"""
    
    # Binance Config
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET: str = os.getenv('BINANCE_API_SECRET')
    
    # Reddit Config
    REDDIT_CLIENT_ID: str = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET: str = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT: str = os.getenv('REDDIT_USER_AGENT', 'CryptoVolt/3.0')
    
    # Discord Config
    DISCORD_WEBHOOK_URL: str = os.getenv('DISCORD_WEBHOOK_URL')
    DISCORD_BOT_TOKEN: str = os.getenv('DISCORD_BOT_TOKEN')
    
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///./cryptovolt.db')
    
    # System
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    PAPER_TRADING_MODE: bool = os.getenv('PAPER_TRADING_MODE', 'True').lower() == 'true'
    
    # Sentiment Analysis
    USE_FINBERT: bool = os.getenv('USE_FINBERT', 'False').lower() == 'true'
    SENTIMENT_CACHE_TTL: int = int(os.getenv('SENTIMENT_CACHE_TTL', '600'))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

---

## Testing the Integration

### 1. Test Sentiment API Endpoint

```bash
# Get Bitcoin sentiment
curl http://localhost:8000/api/sentiment/bitcoin

# Get custom coin sentiment
curl http://localhost:8000/api/sentiment/cardano

# Analyze custom text
curl -X POST http://localhost:8000/api/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin looking bullish!"}'

# Check health
curl http://localhost:8000/api/sentiment/health
```

### 2. Unit Tests

**File**: `tests/test_sentiment.py`

```python
import pytest
from sentiment.sentiment_service import SentimentService

@pytest.fixture
def sentiment_service():
    return SentimentService()

def test_analyze_bitcoin(sentiment_service):
    """Test Bitcoin sentiment analysis"""
    result = sentiment_service.get_bitcoin_sentiment()
    
    assert result.final_sentiment in ['BULLISH', 'BEARISH', 'NEUTRAL']
    assert -1.0 <= result.final_score <= 1.0
    assert 0.0 <= result.final_confidence <= 1.0
    assert result.total_samples > 0

def test_analyze_text(sentiment_service):
    """Test single text analysis"""
    result = sentiment_service.analyze_text("Bitcoin to the moon! 🚀")
    
    assert result['compound'] > 0.5
    assert not result['is_sarcastic']
    assert result['confidence'] > 0.5

def test_sarcasm_detection(sentiment_service):
    """Test sarcasm detection"""
    result = sentiment_service.analyze_text("Great job crashing the market!")
    
    assert result['is_sarcastic'] == True
    assert result['compound'] < 0
```

---

## Integration with Decision Engine

### Fusing Sentiment with Trading Signals

**File**: `backend/app/trading/decision_engine.py`

```python
from sentiment.sentiment_service import get_sentiment_service

class HybridDecisionEngine:
    """Fuse technical signals with ML and sentiment analysis"""
    
    def __init__(self):
        self.sentiment_service = get_sentiment_service()
    
    def make_trading_decision(
        self,
        symbol: str,
        technical_signal: str,  # BUY, SELL, HOLD
        ml_confidence: float,
        price: float
    ) -> tuple[str, float]:
        """
        Make final trading decision fusing multiple signals.
        
        Args:
            symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
            technical_signal: Signal from technical indicators
            ml_confidence: Confidence from ML model (0-1)
            price: Current price
        
        Returns:
            (final_signal, final_confidence)
        """
        
        # Get sentiment
        coin = symbol.replace('USDT', '').lower()
        sentiment = self.sentiment_service.analyze_coin(coin)
        
        # Initialize final decision
        final_signal = technical_signal
        final_confidence = ml_confidence
        
        # Apply sentiment veto
        if sentiment.final_score < -0.3:  # Strong bearish sentiment
            if technical_signal == 'BUY':
                # Don't buy in strong bearish sentiment
                final_signal = 'HOLD'
                final_confidence = 0.3
            else:
                # Increase confidence in SELL during bearish
                final_confidence = min(final_confidence * 1.5, 1.0)
        
        # Sentiment alignment bonus
        if technical_signal == 'BUY' and sentiment.final_sentiment == 'BULLISH':
            # Both aligned, increase confidence
            final_confidence = min(final_confidence * 1.2, 1.0)
        
        if technical_signal == 'SELL' and sentiment.final_sentiment == 'BEARISH':
            # Both aligned, increase confidence
            final_confidence = min(final_confidence * 1.2, 1.0)
        
        return final_signal, final_confidence
```

---

## Monitoring & Troubleshooting

### Check Reddit OAuth Status
```bash
# Test Reddit authentication
d:\cryptovolt\venv\Scripts\python.exe -c "
from sentiment.analyzer import EnhancedCryptoSentimentAnalyzer
analyzer = EnhancedCryptoSentimentAnalyzer(reddit_config={
    'client_id': 'YOUR_ID',
    'client_secret': 'YOUR_SECRET',
    'user_agent': 'CryptoVolt/3.0'
})
print('✅ Reddit OAuth configured successfully!')
"
```

### Monitor API Performance
```python
import time
from sentiment.sentiment_service import get_sentiment_service

service = get_sentiment_service()

start = time.time()
sentiment = service.get_bitcoin_sentiment()
elapsed = time.time() - start

print(f"Analysis completed in {elapsed:.1f}s")
print(f"Sentiment: {sentiment.final_sentiment}")
print(f"Confidence: {sentiment.final_confidence:.2%}")
print(f"Samples: {sentiment.total_samples}")
```

### Check Cache Effectiveness
```python
# Enable debug logging to see cache hits
import logging
logging.getLogger('sentiment.analyzer').setLevel(logging.DEBUG)

# First call - cache miss
result1 = service.get_bitcoin_sentiment()  # ~100-150s

# Second call - cache hit
result2 = service.get_bitcoin_sentiment()  # ~1-5s (cached)
```

---

## Production Deployment

### 1. Update Requirements
```bash
d:\cryptovolt\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with actual credentials
```

### 3. Verify Setup
```bash
# Run tests
pytest tests/test_sentiment.py -v

# Check API health
curl http://localhost:8000/api/sentiment/health

# Get sentiment
curl http://localhost:8000/api/sentiment/bitcoin
```

### 4. Start Server
```bash
cd backend
d:\cryptovolt\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

---

## Summary of Files Created

✅ **Analysis Documents**:
- `API_INTEGRATION_GUIDE.md` - Comprehensive API documentation
- `SENTIMENT_ANALYZER_ANALYSIS.md` - Detailed code analysis
- `.env.example` - Environment configuration template

✅ **Code Files to Create**:
- `backend/app/sentiment/analyzer.py` - Main analyzer class (provided)
- `backend/app/sentiment/sentiment_models.py` - Pydantic models
- `backend/app/sentiment/sentiment_service.py` - Business logic
- `backend/app/api/sentiment.py` - FastAPI routes
- Update `backend/app/core/config.py` - Configuration
- Update `backend/app/main.py` - Register routes
- Update `backend/requirements.txt` - Dependencies
- Update `tests/test_sentiment.py` - Unit tests

**Total Implementation Time**: ~2-3 hours  
**Testing Time**: ~1 hour  
**Integration Testing**: ~1 hour

---

**Status**: ✅ Ready to implement!
