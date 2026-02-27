# CryptoVolt

**AI-Powered Algorithmic Trading Platform with Sentiment Analysis**

CryptoVolt is a research-grade prototype that integrates machine learning models and multi-source sentiment analysis to make intelligent automated trading decisions in cryptocurrency markets. The system fuses technical indicators, sentiment data, and AI predictions with risk-aware veto mechanisms for safer algorithmic trading.

## Overview

CryptoVolt addresses a critical gap in existing trading systems by:
- **Fusing multiple signals**: Technical indicators (rules) + ML predictions + Sentiment analysis
- **Risk-aware decision making**: Sentiment-based veto mechanism that halts trading during high-risk periods
- **Reproducibility**: Full versioning of models, data, configurations, and decision provenance
- **Paper trading**: Safe testing mode by default for all experimentation
- **Academic rigor**: Comprehensive logging and evaluation pipeline for thesis validation

## Key Features

### 🤖 Hybrid Decision Engine
- Combines rule-based technical analysis with machine learning confidence scores
- Sentiment-aware risk veto system
- Explainable trade provenance (audit trail of decisions)

### 📊 Data Integration
- Real-time market data from Binance Futures API
- Multi-source sentiment ingestion (News, Reddit, Twitter/X)
- Technical indicators: EMA, BB, RSI, MACD, Volume analysis
- Documented sampling plan (~300 items/window: 150 Reddit + 150 News)

### 🧠 ML Models
- **XGBoost Classifier**: Binary signal generation (Buy/Sell detection)
- **LSTM Forecaster**: Time-series price prediction
- Model Registry: Version management with metrics tracking
- Feature Store: Reproducible feature engineering

### 💰 Trading Execution
- **Paper Trading Mode** (default): Simulated execution without real funds
- Position management and P&L tracking
- Risk controls: Position limits, max daily loss, stop-loss rules
- Order lifecycle management

### 📱 User Interface
- **PWA Dashboard**: Progressive Web App for desktop & mobile
- Real-time signal monitoring
- Strategy configuration and backtesting
- Performance analytics and reporting

### 🔔 Alerts & Monitoring
- Discord webhook integration for real-time notifications
- Trade execution alerts
- System health monitoring
- Sentiment trend alerts

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI (Python 3.9+) |
| **Frontend** | React 18 (PWA) |
| **Database** | PostgreSQL 15 |
| **Cache** | Redis 7 |
| **ML Framework** | TensorFlow/PyTorch + XGBoost |
| **Deployment** | Docker & Docker Compose |
| **APIs** | Binance Futures, News APIs, Reddit, Twitter/X |

## Project Structure

```
CryptoVolt/
├── backend/                 # FastAPI Application
│   ├── app/
│   │   ├── api/            # API Routes (users, strategies, signals, etc.)
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic services
│   │   ├── trading/        # Decision engine & risk management
│   │   ├── sentiment/      # Sentiment analysis service
│   │   ├── ml/            # ML models & registry
│   │   ├── data/          # Data ingestion & features
│   │   ├── core/          # Config, database, auth
│   │   └── main.py        # FastAPI app initialization
│   ├── requirements.txt    # Python dependencies
│   ├── .env              # Environment variables
│   └── Dockerfile        # Backend Docker image
├── frontend/              # React PWA
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── tests/                # Test suites
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/                 # Documentation
│   ├── SRS.md           # Requirements Specification
│   ├── ARCHITECTURE.md   # System Architecture
│   └── API_DOCS.md      # API Documentation
├── docker-compose.yml    # Multi-container orchestration
└── README.md            # This file
```

## Getting Started

### Prerequisites
- Docker & Docker Compose
- OR: Python 3.9+, Node.js 18+, PostgreSQL 15, Redis 7

### Option 1: Docker Compose (Recommended)

1. **Clone and navigate to project**
   ```bash
   cd d:\CryptoVolt
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Services will be available at:**
   - Backend API: http://localhost:8000
   - Frontend PWA: http://localhost:3000
   - API Health: http://localhost:8000/health

4. **View logs**
   ```bash
   docker-compose logs -f backend
   ```

### Option 2: Local Development

#### Backend Setup

1. **Create Python virtual environment**
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Initialize database** (with PostgreSQL running)
   ```bash
   python -c "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"
   ```

5. **Start API server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

#### Frontend Setup

1. **Install Node dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**
   ```bash
   npm run dev
   ```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /` - Root endpoint (API info)

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout

### Users
- `POST /api/v1/users/register` - Register new user
- `GET /api/v1/users/{user_id}` - Get user profile

### Trading Strategies
- `POST /api/v1/strategies/` - Create strategy
- `GET /api/v1/strategies/{strategy_id}` - Get strategy
- `PUT /api/v1/strategies/{strategy_id}` - Update strategy

### Market Data
- `GET /api/v1/market/candles/{symbol}` - Get candlestick data
- `GET /api/v1/market/data/{symbol}` - Get market data

### Sentiment Analysis
- `GET /api/v1/sentiment/score/{symbol}` - Get sentiment score
- `GET /api/v1/sentiment/data/{symbol}` - Get sentiment data

### Signals & Trades
- `POST /api/v1/signals/` - Generate signal
- `GET /api/v1/signals/{symbol}` - Get signals for symbol
- `POST /api/v1/trades/` - Execute trade
- `GET /api/v1/trades/{symbol}` - Get trades for symbol

### ML Models
- `POST /api/v1/models/` - Register new model
- `GET /api/v1/models/{model_id}` - Get model details

### Alerts
- `POST /api/v1/alerts/` - Create alert
- `GET /api/v1/alerts/{user_id}` - Get user alerts

## Configuration

### Core Settings (`.env` file)

```env
# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Database (default: PostgreSQL on localhost)
DATABASE_URL=postgresql://postgres:password@localhost:5432/cryptovolt

# Binance API (leave empty for testnet)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=True  # Use testnet by default

# Trading
PAPER_TRADING_MODE=True  # Paper trading enabled by default
MAX_POSITION_SIZE=1000
DEFAULT_STOP_LOSS_PCT=2.0

# Sentiment
DISCORD_WEBHOOK_URL=your_webhook

# Redis
REDIS_URL=redis://localhost:6379
```

## Development Workflow

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# All tests with coverage
pytest tests/ --cov=app
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migration
alembic upgrade head
```

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/
pylint app/

# Type checking
mypy app/
```

## System Architecture

The CryptoVolt system consists of:

1. **Data Layer**: Binance API + News/Social APIs → Raw data ingestion
2. **Feature Engineering**: Transform raw data → Technical indicators + Sentiment features
3. **ML Layer**: XGBoost (classification) + LSTM (forecasting) models
4. **Decision Engine**: Fuse ML output + rules + sentiment → Trading signal
5. **Risk Layer**: Validate trades against position/exposure limits
6. **Execution Layer**: Paper trading execution + order management
7. **UI Layer**: PWA dashboard + Discord alerts

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed diagrams and component descriptions.

## Key Concepts

### Hybrid Decision Engine
Combines three signals with configured weights (default: ML=60%, Rules=30%, Sentiment=10%):
- **Rules**: Traditional technical indicators (EMA, BB, RSI)
- **ML**: XGBoost classifier confidence score
- **Sentiment**: Market mood from news/social (-1 to +1)
- **Risk Veto**: Stops trading if sentiment + volatility indicate danger

### Signal Types
- **BUY** (Confidence 0.6-1.0): Fused score confirms uptrend
- **SELL** (Confidence 0.0-0.4): Fused score confirms downtrend
- **HOLD** (Confidence ~0.5): Inconclusive conditions or risk veto

### Paper Trading
Simulates real execution using:
- Real market prices from Binance
- Realistic slippage/fees simulation
- Position tracking and P&L calculation
- No real funds at risk

### Model Versioning
All models stored in Model Registry with:
- Version ID + creation timestamp
- Training metrics (accuracy, precision, recall, AUC)
- Configuration parameters
- Training data slices for reproducibility

## Testing Strategy

Per SRS Section 8 (Test Cases):

1. **Functional Tests (TC-01 to TC-13)**
   - User registration and strategy creation
   - Data ingestion and signal generation
   - Trade execution and P&L tracking
   - Alert delivery

2. **Non-Functional Tests (NFT-01 to NFT-08)**
   - Performance (latency <2s for signal generation)
   - Scalability (50+ symbols, 20+ concurrent users)
   - Security (encrypted credentials, TLS)
   - Accuracy (ML precision/recall targets)

3. **Evaluation**
   - Backtesting on 3+ years historical data
   - Live paper trading with same model/config
   - Sensitivity analysis across parameters

## Backtesting

Run historical strategy evaluation:

```python
from app.ml.backtester import Backtester

backtester = Backtester(
    symbol="BTCUSDT",
    start_date="2023-01-01",
    end_date="2024-01-01",
)

results = backtester.run(
    strategy_id=1,
    model_id=1,
)

print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"CAGR: {results['cagr']:.2%}")
```

## Monitoring & Logging

- Central logging to file: `./logs/cryptovolt.log`
- API request/response logging
- Trade execution audit trail
- Model performance tracking
- Discord webhook notifications for alerts

## Reproducibility

CryptoVolt is designed for academic reproducibility:
- **Versioned Models**: All models tracked in registry
- **Archived Sentiment**: Historical sentiment data preserved
- **Configuration Snapshots**: Strategy params versioned
- **Decision Provenance**: Every trade traces back to inputs
- **Environment Export**: Docker ensures consistent runtime

## Contributing

Development best practices:
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Use conventional commits
- Create feature branches

## Academic Use

CryptoVolt is intended for research and academic purposes:
- **Paper Trading**: Safe by default, no real funds at risk
- **Evaluation**: Metrics suitable for thesis validation
- **Documentation**: Clear for academic assessment
- **Extensibility**: Modular architecture for future research

## Limitations & Scope

**Included:**
- Binance Futures perpetual contracts only
- Paper trading simulation
- Multi-timeframe technical analysis
- English-language sentiment analysis
- Single-user deployment

**Not Included:**
- User fund custody/wallet management
- Multi-user billing system
- Production high-availability SLAs
- Real-money trading (by default)
- Multi-exchange arbitrage
- On-chain trading

## Security Considerations

- All API credentials encrypted at rest
- TLS 1.2+ for external communication
- HMAC authentication for Binance API
- Rate limiting and backoff strategies
- Input validation on all endpoints
- Regular dependency updates

## Support & Documentation

- **API Documentation**: [API_DOCS.md](docs/API_DOCS.md)
- **Architecture Guide**: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Setup Guide**: [SETUP.md](docs/SETUP.md)
- **SRS Document**: [SRS.md](docs/SRS.md)

## License

This project is provided for academic and research purposes. See LICENSE file for details.

## References

- IEEE Std 830-1998: SRS Guidelines
- Binance API Documentation: https://binance-docs.github.io/apidocs/futures/
- XGBoost Paper: Chen & Guestrin (2016)
- LSTM Fundamentals: Hochreiter & Schmidhuber (1997)
- Sentiment Analysis in Finance: Bollen et al. (2011)

## Acknowledgments

This project integrates feedback from:
- Faculty supervisors and academic committee
- Peer reviewers and testers
- Trading research community

---

**Last Updated**: February 2026  
**Status**: Active Development  
**Version**: 1.0.0
