# CryptoVolt Project Instructions

## Project Overview
CryptoVolt is an AI-based algorithmic trading system integrating machine learning and sentiment analysis for cryptocurrency trading on Binance Futures.

## Technology Stack
- **Backend**: FastAPI (Python 3.9+)
- **Frontend**: React PWA
- **Database**: PostgreSQL
- **ML Models**: XGBoost, LSTM
- **APIs**: Binance Futures, News/Sentiment sources
- **Deployment**: Docker, Docker Compose

## Development Setup
1. Configure Python environment (venv recommended)
2. Install backend dependencies from `backend/requirements.txt`
3. Set up PostgreSQL database
4. Configure environment variables
5. Set up frontend (Node.js, npm/yarn)

## Key Modules
- Data Ingestion Pipeline (market data, sentiment data)
- Feature Engineering (technical indicators, sentiment scoring)
- ML Models (XGBoost classifier, LSTM forecaster)
- Decision Engine (hybrid rule + ML logic)
- Execution Layer (paper trading simulation)
- Monitoring UI (PWA dashboard)

## Project Structure
```
CryptoVolt/
├── backend/           # FastAPI backend
├── frontend/          # React PWA
├── docs/              # Documentation
├── tests/             # Test suites
├── docker-compose.yml # Multi-container setup
└── README.md          # Project readme
```
