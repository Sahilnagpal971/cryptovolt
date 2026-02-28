# PostgreSQL Backend Setup Documentation

## Overview
CryptoVolt uses PostgreSQL as its primary database for storing:
- User accounts and authentication
- Trading strategies
- Market data (OHLCV candles)
- Sentiment analysis results
- Trading signals
- Trade execution records
- ML model metadata
- System alerts

## Database Configuration

### Connection Details
- **Database URL Format**: `postgresql://username:password@host:port/database_name`
- **Default Local Setup**: `postgresql://postgres:password@localhost:5432/cryptovolt`
- **Docker Setup**: `postgresql://postgres:password@postgres:5432/cryptovolt`

### Connection Parameters (Configurable via Environment Variables)
All database configuration is set in the `.env` file:

```
DATABASE_URL=postgresql://postgres:password@localhost:5432/cryptovolt
REDIS_URL=redis://localhost:6379
```

## Setup Methods

### Method 1: Docker Compose (Recommended)
The easiest way to set up the complete stack with PostgreSQL:

```bash
# Navigate to project root
cd d:\CryptoVolt

# Start all services (PostgreSQL, Redis, FastAPI, React)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f postgres
```

**Features:**
- Automatic database initialization
- Data persistence in volumes
- Network isolation
- Production-ready configuration

### Method 2: Local PostgreSQL Installation
For development without Docker:

1. **Install PostgreSQL** (Windows):
   - Download from: https://www.postgresql.org/download/windows/
   - Use default password: `postgres` during installation
   
2. **Create Database**:
   ```bash
   # Using psql (PostgreSQL CLI)
   psql -U postgres -c "CREATE DATABASE cryptovolt OWNER postgres;"
   ```

3. **Update Connection String**:
   - Edit `.env` file
   - Set: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cryptovolt`

4. **Initialize Database**:
   ```bash
   cd d:\CryptoVolt\backend
   python manage_db.py init
   ```

### Method 3: Cloud Database
For production deployments:

```
# AWS RDS PostgreSQL Example
DATABASE_URL=postgresql://admin:password@cryptovolt-db.c4a8zq3x.us-east-1.rds.amazonaws.com:5432/cryptovolt

# Azure Database for PostgreSQL Example
DATABASE_URL=postgresql://admin@servername:password@servername.postgres.database.azure.com:5432/cryptovolt
```

## Database Initialization

### Automatic Initialization
The application automatically initializes the database on startup:
```python
# In app/main.py
if verify_database_connection():
    init_db()  # Creates all tables if they don't exist
```

### Manual Initialization
For manual control, use the database management script:

```bash
cd d:\CryptoVolt\backend

# Initialize database (create tables)
python manage_db.py init

# Check database statistics
python manage_db.py stats

# Verify connection
python manage_db.py verify

# Seed with sample data (admin user)
python manage_db.py seed

# Clean old records (>30 days sentiment, >90 days market data)
python manage_db.py clean

# Reset database (DELETE ALL DATA - CAREFUL!)
python manage_db.py reset
```

---

## Database Schema

### Core Tables

#### 1. **users**
Stores user account information
```sql
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. **trading_strategies**
Stores trading strategy configurations
```sql
CREATE TABLE trading_strategies (
    strategy_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB,  -- Strategy parameters (flexible)
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. **market_data**
Stores historical market data (OHLCV)
```sql
CREATE TABLE market_data (
    data_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) INDEX,
    timestamp TIMESTAMP INDEX,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume FLOAT,
    source VARCHAR(50),  -- 'binance', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. **sentiment_data**
Stores sentiment analysis results
```sql
CREATE TABLE sentiment_data (
    sentiment_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) INDEX,
    source VARCHAR(50) INDEX,  -- 'reddit', 'twitter', 'news'
    text TEXT,
    sentiment_score FLOAT,  -- -1.0 to 1.0
    timestamp TIMESTAMP INDEX,
    extra_data JSONB,  -- Additional metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5. **signals**
Stores trading signals from models
```sql
CREATE TABLE signals (
    signal_id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES trading_strategies(strategy_id),
    model_id INTEGER REFERENCES models(model_id),
    symbol VARCHAR(20),
    signal_type VARCHAR(10),  -- 'BUY', 'SELL', 'HOLD'
    confidence FLOAT,  -- 0.0 to 1.0
    timestamp TIMESTAMP INDEX,
    extra_data JSONB,  -- Additional metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 6. **trades**
Stores executed trades
```sql
CREATE TABLE trades (
    trade_id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(signal_id),
    symbol VARCHAR(20),
    trade_type VARCHAR(10),  -- 'BUY', 'SELL'
    price FLOAT,
    quantity FLOAT,
    status VARCHAR(20),  -- 'PENDING', 'EXECUTED', 'FAILED'
    is_paper_trade BOOLEAN DEFAULT true,
    pnl FLOAT,  -- Profit/Loss
    timestamp TIMESTAMP INDEX,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 7. **models**
Stores ML model metadata
```sql
CREATE TABLE models (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    model_type VARCHAR(50),  -- 'xgboost', 'lstm'
    version VARCHAR(20),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    auc FLOAT,
    trained_on TIMESTAMP,
    extra_data JSONB,  -- Model parameters, features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 8. **alerts**
Stores system and user alerts
```sql
CREATE TABLE alerts (
    alert_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    alert_type VARCHAR(50),  -- 'SIGNAL', 'TRADE', 'SYSTEM'
    message TEXT,
    is_read BOOLEAN DEFAULT false,
    timestamp TIMESTAMP INDEX,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Database Operations

### Python ORM Usage
Using SQLAlchemy ORM for database operations:

```python
from app.core.database import SessionLocal
from app.models.database import User, MarketData
from app.services.database_service import DatabaseService

# Create session
db = SessionLocal()

# Create user
new_user = DatabaseService.create(
    db, 
    User, 
    username="trader",
    email="trader@cryptovolt.local",
    password_hash="hashed_password"
)

# Query users
users = db.query(User).filter(User.is_active == True).all()

# Update user
DatabaseService.update(db, User, new_user.user_id, is_active=False)

# Delete user
DatabaseService.delete(db, User, new_user.user_id)

# Close session
db.close()
```

### Direct SQL Queries
For complex operations not covered by ORM:

```python
from sqlalchemy import text

with engine.connect() as connection:
    result = connection.execute(text("""
        SELECT symbol, AVG(close_price) as avg_price
        FROM market_data
        GROUP BY symbol
        ORDER BY avg_price DESC
    """))
    for row in result:
        print(row)
```

---

## Database Utilities

### DatabaseService Class
Generic CRUD operations for any model:

```python
from app.services.database_service import DatabaseService, QueryHelper, DatabaseStats

# CRUD Operations
DatabaseService.create(db, ModelClass, **kwargs)
DatabaseService.read(db, ModelClass, id)
DatabaseService.update(db, ModelClass, id, **kwargs)
DatabaseService.delete(db, ModelClass, id)
DatabaseService.list_all(db, ModelClass, skip=0, limit=100)
DatabaseService.bulk_create(db, ModelClass, [objects...])

# Query Helpers
QueryHelper.filter_by_symbol(db, ModelClass, "BTCUSDT")
QueryHelper.filter_by_date_range(db, ModelClass, start, end)
QueryHelper.search_text(db, ModelClass, "search_term", ["column1", "column2"])

# Statistics
DatabaseStats.get_table_row_count(db, ModelClass)
DatabaseStats.get_all_table_stats(db, [Model1, Model2, ...])
```

---

## Performance Optimization

### Indexes
Key columns are indexed for fast queries:
- `users.username`, `users.email`
- `market_data.symbol`, `market_data.timestamp`
- `sentiment_data.symbol`, `sentiment_data.source`
- `signals.timestamp`
- `trades.timestamp`

### Connection Pooling
SQLAlchemy connection pooling is configured:
```python
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.DEBUG,  # Log SQL queries in debug mode
)
```

### Query Optimization
- Use `.select()` with filters instead of fetching all rows
- Implement pagination with `skip` and `limit`
- Use batch operations for bulk inserts
- Monitor slow queries with `echo=True`

---

## Backup & Maintenance

### Backup (Docker)
```bash
# Create backup
docker exec cryptovolt_db pg_dump -U postgres cryptovolt > backup.sql

# Restore backup
docker exec -i cryptovolt_db psql -U postgres cryptovolt < backup.sql
```

### Backup (Local PostgreSQL)
```bash
# Create backup
pg_dump -U postgres cryptovolt > backup.sql

# Restore backup
psql -U postgres cryptovolt < backup.sql
```

### Database Cleanup
Remove old data to save disk space:
```bash
python manage_db.py clean
```

---

## Troubleshooting

### Connection Issues
```python
# Test connection
from app.core.init_db import verify_database_connection
result = verify_database_connection()
# True = connected, False = failed
```

### Common Errors

1. **"could not connect to server"**
   - Check if PostgreSQL is running
   - Verify host, port, and credentials in DATABASE_URL
   - Check firewall settings

2. **"database 'cryptovolt' does not exist"**
   - Run: `python manage_db.py init`
   - Or manually: `createdb -U postgres cryptovolt`

3. **"permission denied"**
   - Check user credentials in DATABASE_URL
   - Verify PostgreSQL user exists and has permissions
   - Run: `psql -U postgres -c "GRANT ALL ON DATABASE cryptovolt TO postgres;"`

4. **"metadata is reserved"**
   - Fixed: Renamed `metadata` columns to `extra_data` in models

### View Database Logs
```bash
# Docker
docker-compose logs postgres

# Local PostgreSQL
# Logs usually in: C:\Program Files\PostgreSQL\15\data\log\
```

---

## Production Deployment Checklist

- [ ] Use strong password (not "password")
- [ ] Enable SSL/TLS for database connections
- [ ] Set up automated backups
- [ ] Monitor database performance
- [ ] Use connection pooling (pgbouncer for high-load)
- [ ] Follow PostgreSQL security best practices
- [ ] Regular data cleanup and archival
- [ ] Test disaster recovery procedures
- [ ] Set up monitoring and alerting
- [ ] Document backup and restore procedures

---

## Next Steps

1. **Start Services**: 
   ```bash
   docker-compose up -d
   ```

2. **Verify Setup**:
   ```bash
   docker-compose logs postgres
   ```

3. **Access API**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Frontend: http://localhost:3000

4. **Interact with Database**:
   - Use provided Python scripts
   - Or connect via pgAdmin/DBeaver
   - Or use FastAPI's interactive documentation

---

## Resources

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- SQLAlchemy ORM: https://docs.sqlalchemy.org/
- Docker PostgreSQL: https://hub.docker.com/_/postgres
- pgAdmin (Web Interface): https://www.pgadmin.org/
