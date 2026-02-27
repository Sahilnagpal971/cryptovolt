# CryptoVolt Setup Guide

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Docker**: 20.10+ and Docker Compose 2.0+
- **OR** for local development:
  - Python 3.9+
  - Node.js 18+
  - PostgreSQL 15+
  - Redis 7+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 20GB for data and models
- **Internet**: Stable connection for APIs

## Quick Start (Docker)

### 1. Clone Repository
```bash
git clone <repository-url>
cd CryptoVolt
```

### 2. Create Environment File
```bash
cp backend/.env.example backend/.env
# Edit backend/.env as needed
```

### 3. Start All Services
```bash
docker-compose up -d
```

### 4. Verify Services
```bash
# Check API
curl http://localhost:8000/health

# Check frontend
Open http://localhost:3000 in browser
```

### 5. Initialize Database
```bash
docker-compose exec backend python -c \
  "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"
```

## Local Development Setup

### Backend Setup (Linux/macOS)

1. **Create virtual environment**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database setup**
   ```bash
   # PostgreSQL must be running
   createdb -U postgres cryptovolt
   python -c "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"
   ```

4. **Start API server**
   ```bash
   uvicorn app.main:app --reload
   ```

### Backend Setup (Windows)

1. **Create virtual environment**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Database setup**
   ```powershell
   # PostgreSQL must be running and accessible
   python -c "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"
   ```

4. **Start API server**
   ```powershell
   uvicorn app.main:app --reload
   ```

### Frontend Setup (All OS)

1. **Install dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**
   ```bash
   npm run dev
   ```

3. **Open browser**
   ```
   http://localhost:3000
   ```

## Environment Configuration

### PostgreSQL Setup

```bash
# Create database
createdb cryptovolt

# Create user (if needed)
createuser -P cryptovolt_user
```

### Redis Setup

```bash
# Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# OR system installation
redis-server
```

## Binance API Configuration

### For Paper Trading (Recommended)
Leave `BINANCE_API_KEY` and `BINANCE_API_SECRET` empty in `.env`:
- Uses Binance Testnet automatically
- No real funds required
- Safe for experiments

### For Live Testing
1. Create Binance account
2. Generate API Key: https://www.binance.com/en/account/api-management
3. Enable Futures trading
4. Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `.env`
5. Keep `BINANCE_TESTNET=false` for production API

**⚠️ WARNING**: Real API keys with live trading are not recommended for research. Use paper trading mode.

## Sentiment Analysis API Setup

### News API
1. Sign up: https://newsapi.org
2. Get free API key
3. Add to `.env`: `NEWS_API_KEY=your_key`

### Reddit API
1. Create app: https://www.reddit.com/prefs/apps
2. Get credentials
3. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   ```

### Twitter/X API
1. Apply for access: https://developer.twitter.com
2. Get API keys
3. Will be added to future versions

## Discord Webhook Setup

1. **Create Discord Server** (if needed)
2. **Create Webhook**:
   - Server Settings → Integrations → Webhooks
   - Create new webhook
   - Copy webhook URL
3. **Add to `.env`**:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

## Testing the Setup

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database connection
docker-compose exec backend python -c \
  "from app.core.database import SessionLocal; db = SessionLocal(); print('✓ DB Connected')"

# Redis connection
docker exec cryptovolt_redis redis-cli ping
```

### Create Test Strategy

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/strategies/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "name": "Test Strategy",
    "description": "Simple test",
    "parameters": {"threshold": 0.6}
  }'
```

### Run Test Suite

```bash
# Backend tests
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## Troubleshooting

### Database Connection Error
```
Error: could not connect to server: Connection refused
```
**Solution**: Ensure PostgreSQL is running
```bash
# Docker
docker-compose up postgres

# Local
postgres -D /var/lib/postgresql/data
```

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change port in `.env` or kill existing process
```bash
# Find process on port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux

# Kill process
kill -9 <PID>
```

### Redis Connection Error
**Solution**: Ensure Redis is running
```bash
docker-compose exec backend redis-cli ping
```

### CORS Errors in Frontend
**Solution**: Update `CORS_ORIGINS` in `backend/app/core/config.py`:
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",  # If using Vite
]
```

## Performance Tuning

### PostgreSQL
- Connection pooling: Configured via SQLAlchemy
- Index optimization: Add indexes to frequently queried columns
- Vacuum: `VACUUM ANALYZE;`

### Redis
- Memory limit: Set `maxmemory` in redis.conf
- Eviction policy: `maxmemory-policy allkeys-lru`

### Python/FastAPI
- Worker processes: Scale based on CPU cores
- Batching: Process market data in batches
- Caching: Use Redis for feature cache

## Monitoring

### Logs
```bash
# Backend logs
docker-compose logs -f backend

# All services
docker-compose logs -f
```

### Database
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d cryptovolt

# List tables
\dt

# View recent trades
SELECT * FROM trades ORDER BY created_at DESC LIMIT 10;
```

### Redis
```bash
# Monitor commands
docker-compose exec redis redis-cli MONITOR

# Check memory
docker-compose exec redis redis-cli INFO memory
```

## Scaling for Production

⚠️ **Note**: CryptoVolt is designed for research, not production. For deployment:

1. **Use managed services**:
   - Cloud PostgreSQL (AWS RDS, Google Cloud SQL)
   - Managed Redis (AWS ElastiCache)
   - Load balancer (AWS ALB, Google HTTPS LB)

2. **Security hardening**:
   - Enable TLS/SSL
   - Use secrets manager
   - Network segmentation
   - WAF rules

3. **Monitoring & observability**:
   - CloudWatch, Stackdriver
   - APM: New Relic, DataDog
   - Log aggregation: ELK, Splunk

4. **High availability**:
   - Multi-region deployment
   - Database replication
   - Failover procedures

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Check [API_DOCS.md](API_DOCS.md) for endpoint details
3. Review [SRS.md](SRS.md) for requirements
4. Start with example strategies in `backend/app/trading/examples/`

## Support

For issues and questions:
- Check logs: `docker-compose logs backend`
- Review docs in `docs/` directory
- Check test files for usage examples
- Review issue tracker

---

**Last Updated**: February 2026
