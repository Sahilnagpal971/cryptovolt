# CryptoVolt Quick Reference Guide

## Common Commands

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f backend   # Backend logs
docker-compose logs -f frontend  # Frontend logs

# Execute command in container
docker-compose exec backend python -c "..."
docker-compose exec postgres psql -U postgres -d cryptovolt

# Rebuild containers
docker-compose build --no-cache
```

### Backend Development

```bash
# Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS/Linux

# Run FastAPI server
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
pytest tests/test_decision_engine.py -v

# Check database
python -c "from app.core.database import SessionLocal; db = SessionLocal(); print('OK')"

# Format code
black app/
isort app/
```

### Frontend Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Format code
npm run format
```

---

## API Endpoints Cheatsheet

### Health
```
GET /health
GET /
```

### Auth
```
POST /api/v1/auth/login
POST /api/v1/auth/logout
```

### Users
```
POST /api/v1/users/register
GET /api/v1/users/{user_id}
```

### Strategies
```
POST /api/v1/strategies/
GET /api/v1/strategies/{strategy_id}
PUT /api/v1/strategies/{strategy_id}
```

### Market Data
```
GET /api/v1/market/candles/{symbol}
GET /api/v1/market/data/{symbol}
```

### Sentiment
```
GET /api/v1/sentiment/score/{symbol}
GET /api/v1/sentiment/data/{symbol}
```

### Signals
```
POST /api/v1/signals/
GET /api/v1/signals/{symbol}
```

### Trades
```
POST /api/v1/trades/
GET /api/v1/trades/{symbol}
```

### Models
```
POST /api/v1/models/
GET /api/v1/models/{model_id}
```

### Alerts
```
POST /api/v1/alerts/
GET /api/v1/alerts/{user_id}
```

---

## File Structure Quick Map

```
Backend Core Logic:
├── app/core/config.py          # Settings
├── app/core/database.py        # DB connection
├── app/models/database.py      # ORM models
├── app/trading/decision_engine.py    # Main logic
└── app/sentiment/analyzer.py   # Sentiment

API Routes:
├── app/api/health.py
├── app/api/auth.py
├── app/api/strategies.py
├── app/api/market_data.py
├── app/api/sentiment.py
├── app/api/signals.py
├── app/api/trades.py
├── app/api/models.py
└── app/api/alerts.py

ML Pipeline:
├── app/data/ingestion.py       # Data fetching
├── app/ml/models.py            # ML models
└── app/sentiment/analyzer.py   # Sentiment

Frontend:
├── src/main.jsx                # Entry
├── src/App.jsx                 # Main
└── src/index.css               # Styles

Config:
├── backend/.env                # Environment
├── docker-compose.yml          # Docker setup
└── frontend/vite.config.js    # Frontend build

Docs:
├── README.md                   # Overview
├── docs/SETUP.md              # Setup guide
├── docs/SRS.md                # Requirements
└── PROJECT_STATUS.md          # Status
```

---

## Configuration Checklist

### Important Settings in `.env`

```env
# Running locally?
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/cryptovolt

# Using Binance testnet? (Recommended)
BINANCE_TESTNET=True
PAPER_TRADING_MODE=True

# ML paths
MODEL_REGISTRY_PATH=./models
FEATURE_STORE_PATH=./features

# Risk management
MAX_POSITION_SIZE=1000
DEFAULT_STOP_LOSS_PCT=2.0
```

---

## Common Issues & Fixes

### Port Already in Use
```bash
# Find process
netstat -ano | findstr :8000  # Windows

# Kill it
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                 # macOS/Linux

# Or change port in .env
API_PORT=8001
```

### Database Connection Failed
```bash
# Ensure PostgreSQL running
docker-compose up postgres

# Or check connection string in .env
DATABASE_URL=postgresql://user:pass@localhost:5432/cryptovolt
```

### Frontend Can't Connect to API
```bash
# Check API running
curl http://localhost:8000/health

# Check frontend proxy in vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

---

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/my-feature
```

### 2. Make Changes
```bash
# Backend
cd backend
# Edit files in app/
# Add tests in tests/

# Frontend
cd frontend
# Edit files in src/

# Update docs if needed
```

### 3. Test Locally
```bash
# Run tests
pytest tests/ -v

# Check API
curl http://localhost:8000/health

# Check frontend
npm run dev
```

### 4. Commit & Push
```bash
git add .
git commit -m "feat: description"
git push origin feature/my-feature
```

---

## Database Operations

### Connect to PostgreSQL
```bash
docker-compose exec postgres psql -U postgres -d cryptovolt
```

### Common Queries
```sql
-- Check tables
\dt

-- View users
SELECT * FROM users;

-- View trades
SELECT * FROM trades ORDER BY created_at DESC LIMIT 10;

-- View models
SELECT * FROM models;

-- Check record counts
SELECT COUNT(*) FROM trades;
```

### Reset Database
```bash
# Drop everything and recreate
docker-compose exec backend python << 'EOF'
from app.core.database import Base, engine
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print("Database reset!")
EOF
```

---

## Testing Guide

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_decision_engine.py -v
pytest tests/test_decision_engine.py::test_make_decision_buy -v
```

### Run with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

### Run Specific Component
```bash
pytest tests/ -k test_sentiment -v
```

---

## Code Style

### Python (Backend)
- Follow PEP 8
- Use type hints
- Docstrings for functions
- Max line length: 100

```python
def my_function(param1: str, param2: int) -> bool:
    """Function description."""
    return True
```

### JavaScript (Frontend)
- Use functional components
- Use React hooks
- Add comments for complex logic
- Props validation

```jsx
function MyComponent({ data, onUpdate }) {
  const [state, setState] = useState(null);
  return <div>{state}</div>;
}
```

---

## Key Concepts

### Hybrid Decision Engine
```python
from app.trading.decision_engine import DecisionEngine

engine = DecisionEngine({
    "ml_weight": 0.6,
    "rule_weight": 0.3,
    "sentiment_weight": 0.1,
})

decision = engine.make_decision(
    symbol="BTCUSDT",
    ml_prediction={"signal": "BUY", "confidence": 0.8},
    rule_signals={"signal": "BUY", "strength": 0.7},
    sentiment_score=0.5,
    volatility=0.02,
)
```

### Feature Engineering
```python
from app.data.ingestion import FeatureEngine

engine = FeatureEngine()
indicators = engine.calculate_technical_indicators(candles)
features = engine.create_feature_vector(indicators, sentiment, volatility)
```

### Sentiment Analysis
```python
from app.sentiment.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_symbol_sentiment("BTCUSDT", sentiment_data)
```

---

## Resources

- **Binance API**: https://binance-docs.github.io/apidocs/futures/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **Docker**: https://docs.docker.com/

---

## Useful Links

- API at: http://localhost:8000
- API Docs (when implemented): http://localhost:8000/docs
- Frontend at: http://localhost:3000
- Redis: localhost:6379
- PostgreSQL: localhost:5432

---

**Last Updated**: February 26, 2026

Use this guide for quick reference during development!
