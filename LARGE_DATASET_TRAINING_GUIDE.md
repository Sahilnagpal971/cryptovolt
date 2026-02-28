# CryptoVolt Large Dataset & Multi-Coin Model Training Guide

## Overview

This guide walks through fetching large historical datasets for the top 10 reliable cryptocurrencies and training ML models (XGBoost & LSTM) for each coin.

### Top 10 Selected Coins

| Coin | Symbol | Reason |
|------|--------|--------|
| Bitcoin | BTC | Market leader, most liquid |
| Ethereum | ETH | Smart contracts standard, most adopted |
| Solana | SOL | High-speed blockchain, growing ecosystem |
| Ripple | XRP | Payment protocol, institutional adoption |
| Cardano | ADA | Proof-of-stake, sustainable |
| Avalanche | AVAX | Scalable, fast transactions |
| Polygon | MATIC | Layer-2 solution, Ethereum scaling |
| Chainlink | LINK | Oracle infrastructure, critical for DeFi |
| Uniswap | UNI | Leading DEX, DeFi pioneer |
| Polkadot | DOT | Multi-chain framework, interoperability |

## Dataset Information

### Data Collection Strategy
- **Source**: Binance API (fastest, most reliable)
- **Timeframe**: 2 years of hourly data (730 days)
- **Data Points**: ~17,500 candles per coin = **175,000+ total data points**
- **Features**: OHLCV (Open, High, Low, Close, Volume) + Quote Asset Volume, Number of Trades

### Why This Approach?

1. **Large Dataset**: 2 years of hourly data provides sufficient training data
   - Captures multiple market cycles
   - Represents recent market behavior
   - Sufficient for robust ML model training

2. **Hourly Interval** (1h)
   - Balance between granularity and data volume
   - Captures day-trader sentiment + swing patterns
   - More stable than 1m data, more detail than 1d

3. **Multiple Coins**: Diversified training
   - Each coin has unique patterns
   - Enables portfolio-level strategies
   - Validates model generalization

## Setup Instructions

### 1. Prerequisites

```bash
# Ensure you have required Python packages
pip install requests pandas numpy xgboost tensorflow scikit-learn
```

### 2. Fetch Large Datasets

```bash
cd backend

# Run the dataset fetcher (will take 15-30 minutes depending on internet)
python fetch_large_dataset.py
```

#### What This Does:
- Downloads 2 years of 1h candles for all 10 coins
- Fetches ~175,000 total data points
- Handles Binance API rate limits automatically
- Saves CSVs in `./data/raw/`
- Generates `dataset_fetch.log` with detailed progress

#### Output Structure:
```
./data/raw/
├── BTCUSDT_2024-02-28_2026-02-28.csv    (~17,500 rows)
├── ETHUSDT_2024-02-28_2026-02-28.csv
├── SOLUSDT_2024-02-28_2026-02-28.csv
├── ...
└── DOTUSDT_2024-02-28_2026-02-28.csv
```

### 3. Train Multi-Coin Models

```bash
# Train XGBoost + LSTM models for each coin
python train_multi_coin_models.py
```

#### What This Does:
- Loads datasets for all 10 coins
- Calculates 30+ technical indicators per coin
- Trains XGBoost classification model (1% prediction threshold)
- Trains LSTM price forecasting model (60-timestep sequence)
- Registers models in the model registry
- Generates `model_training.log` with training metrics

#### Training Details:

**Technical Indicators (30+)**:
- Moving Averages: SMA(10,20,50), EMA(10,20)
- Momentum: RSI, MACD, Stochastic Oscillator
- Volatility: ATR, Bollinger Bands, Volatility measurements
- Volume: Volume ratio, volume trend
- Advanced: Price position, momentum rates, Stochastic K/D

**XGBoost Model**:
- Type: Binary Classification
- Task: Predict if price will go up 1% in next 6 hours
- Features: 30+ technical indicators
- Training: 80% of data
- Validation: 20% of data (most recent)
- Epochs: 150

**LSTM Model**:
- Type: Sequence-to-Sequence Forecasting
- Task: Predict next hour's closing price
- Sequence Length: 60 timesteps (60 hours of history)
- Architecture: Deep LSTM with dropout
- Batch Size: 32
- Epochs: 100

#### Expected Training Time:
- Per coin: 8-15 minutes (GPU: 2-5 minutes)
- Total for 10 coins: 80-150 minutes
- Can be parallelized for faster training

### 4. Monitor Training Progress

```bash
# In another terminal, watch the logs
tail -f backend/model_training.log
tail -f backend/dataset_fetch.log
```

## Model Output

### Saved Models Location: `./models/`

```
./models/
├── xgboost_btcusdt_v2.0_large.pkl
├── xgboost_ethusdt_v2.0_large.pkl
├── ...
├── lstm_btcusdt_v2.0_large.h5
├── lstm_ethusdt_v2.0_large.h5
└── ... (20 total models)
```

### Model Registry
Each trained model is registered with:
- **Metadata**: Coin symbol, model type, version
- **Metrics**: Accuracy, AUC, MAE values
- **Configuration**: Hyperparameters, feature list

## Expected Performance

### XGBoost Metrics
- Accuracy: 52-58% (better than 50% baseline)
- AUC: 0.55-0.62
- Note: Crypto markets are highly noisy; modest improvement is realistic

### LSTM Metrics
- MAE: 0.005-0.020 (normalized price units)
- RMSE: 0.010-0.035
- Captures trend directions effectively

## Usage After Training

### 1. Load Trained Models in Your Code

```python
from app.ml.models import ModelRegistry

registry = ModelRegistry("./models")

# Load XGBoost model
xgb_model = registry.get_model("xgboost_btcusdt", version="v2.0_large")

# Load LSTM model
lstm_model = registry.get_model("lstm_ethusdt", version="v2.0_large")
```

### 2. Make Predictions

```python
import numpy as np

# XGBoost prediction (needs feature vector)
features = np.array([[sma_10, sma_20, rsi, ...]])  # 30 features
prediction = xgb_model.predict(features)  # 0 or 1

# LSTM prediction (needs 60 timesteps)
prices = np.array([[...60 hours of prices...]])  # Shape: (1, 60, 1)
next_price = lstm_model.predict(prices)
```

### 3. Integration with Trading System

Models can be integrated with:
- Decision Engine (`app/trading/decision_engine.py`)
- API Endpoints (`app/api/signals.py`, `app/api/models.py`)
- Backtesting Engine

## Optimization Tips

### If Running Slowly:

1. **Use GPU** (if available):
   - XGBoost: Set `tree_method="gpu_hist"` (requires cuDF)
   - LSTM: Ensure TensorFlow GPU support is installed

2. **Parallel Training**:
   - Modify `train_multi_coin_models.py` to use multiprocessing
   - Train 2-3 coins simultaneously

3. **Reduce Data**:
   - Use 365 days instead of 730 days
   - Use daily (1d) candles instead of hourly (1h)

### If Fetch Fails:

1. **Rate Limiting**: Binance limits to ~1200 requests/minute
   - Script handles this automatically with delays
   - Increase `wait_time` in `fetch_large_dataset.py` if needed

2. **Connection Issues**: 
   - Check Binance API status: https://www.binance.com/
   - Try running again - script is idempotent

3. **Missing Data**:
   - Some coins may not have 2 years of data
   - Script automatically handles this
   - Minimum 6 months recommended for training

## Next Steps

1. **Run Dataset Fetch** (15-30 min)
   ```bash
   python fetch_large_dataset.py
   ```

2. **Run Model Training** (2-3 hours for all 10 coins)
   ```bash
   python train_multi_coin_models.py
   ```

3. **Integrate Models** into trading decision engine

4. **Backtest** strategies with trained models

5. **Deploy** to production with continuous retraining

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'app'"
```bash
# Ensure you're running from the backend directory
cd backend
python train_multi_coin_models.py
```

### Issue: "Connection error" during dataset fetch
```bash
# Check internet connection and Binance API availability
# Restart the fetch - it will resume from where it stopped
python fetch_large_dataset.py
```

### Issue: Out of Memory during LSTM training
```bash
# Reduce batch size in train_multi_coin_models.py
# Change: batch_size=32 → batch_size=16 or 8
```

### Issue: Models not found after training
```bash
# Check the models/ directory exists and has content
ls -la ./models/
# Should show xgboost_*.pkl and lstm_*.h5 files
```

## Questions or Issues?

Check the logs:
- Dataset logs: `backend/dataset_fetch.log`
- Training logs: `backend/model_training.log`

Logs contain detailed information about what's happening at each step.

---

**Happy Training! 🚀** Your CryptoVolt system will be much more robust with models trained on large, recent datasets for multiple coins!
