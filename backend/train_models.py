"""
Train ML models (XGBoost and LSTM) for CryptoVolt trading system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from app.ml.models import XGBoostClassifier, LSTMForecaster, ModelRegistry
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    """Fetch historical data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    logger.info(f"Fetching {limit} candles for {symbol}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    logger.info(f"Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def calculate_technical_indicators(df):
    """Calculate technical indicators for features"""
    logger.info("Calculating technical indicators...")
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Price position in range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return df


def create_xgboost_labels(df, future_periods=6):
    """Create labels fort XGBoost: 1 if price goes up, 0 otherwise"""
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
    df['label'] = (df['future_return'] > 0.01).astype(int)  # 1% threshold
    return df


def prepare_xgboost_data(df):
    """Prepare feature matrix for XGBoost"""
    feature_columns = [
        'returns', 'sma_10', 'sma_30', 'ema_10', 'rsi',
        'bb_middle', 'bb_upper', 'bb_lower',
        'macd', 'macd_signal', 'volume_ratio', 'volatility', 'price_position'
    ]
    
    # Drop NaN rows
    df_clean = df.dropna()
    
    X = df_clean[feature_columns].values
    y = df_clean['label'].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/val
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"XGBoost data prepared: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val


def prepare_lstm_data(df, sequence_length=60):
    """Prepare sequences for LSTM"""
    # Use close prices
    prices = df['close'].values
    
    X, y = [], []
    for i in range(sequence_length, len(prices)):
        X.append(prices[i-sequence_length:i])
        y.append(prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM [samples, timesteps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    # Split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"LSTM data prepared: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier"""
    logger.info("=" * 60)
    logger.info("Training XGBoost Classifier")
    logger.info("=" * 60)
    
    model = XGBoostClassifier(params={
        "max_depth": 6,
        "eta": 0.1,
        "objective": "binary:logistic",
    })
    
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=100)
    
    logger.info(f"XGBoost Metrics: {metrics}")
    return model, metrics


def train_lstm_model(X_train, y_train, X_val, y_val):
    """Train LSTM forecaster"""
    logger.info("=" * 60)
    logger.info("Training LSTM Forecaster")
    logger.info("=" * 60)
    
    model = LSTMForecaster(sequence_length=60)
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    logger.info(f"LSTM Metrics: {metrics}")
    return model, metrics


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("CryptoVolt ML Model Training Pipeline")
    print("=" * 60 + "\n")
    
    # Initialize model registry
    registry = ModelRegistry("./models")
    
    # Fetch data
    df = fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000)
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    df = create_xgboost_labels(df)
    
    # Train XGBoost
    X_train, y_train, X_val, y_val = prepare_xgboost_data(df)
    xgb_model, xgb_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Register XGBoost model
    xgb_id = registry.register_model(
        model=xgb_model.model,
        model_name="xgboost_classifier",
        model_type="xgboost",
        version="v1.0",
        metrics=xgb_metrics,
        config=xgb_model.params
    )
    print(f"\n✅ XGBoost model registered: {xgb_id}")
    
    # Train LSTM
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm = prepare_lstm_data(df)
    lstm_model, lstm_metrics = train_lstm_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    
    # Register LSTM model
    lstm_id = registry.register_model(
        model=lstm_model.model,
        model_name="lstm_forecaster",
        model_type="lstm",
        version="v1.0",
        metrics=lstm_metrics,
        config={"sequence_length": 60}
    )
    print(f"✅ LSTM model registered: {lstm_id}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nModels saved in: ./models/")
    print(f"XGBoost Accuracy: {xgb_metrics.get('accuracy', 'N/A'):.2%}")
    print(f"LSTM Val MAE: {lstm_metrics.get('val_mae', lstm_metrics.get('final_mae', 0)):.2f}")
    print("\n")


if __name__ == "__main__":
    main()
