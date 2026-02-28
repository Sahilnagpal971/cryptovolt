"""
Standalone multi-coin model training without FastAPI dependencies
Directly trains XGBoost and LSTM models from downloaded datasets
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import os
import pickle
import json
import logging
from typing import Dict, Tuple, List
from datetime import datetime
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('standalone_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top 10 coins to train models for
TOP_10_COINS = {
    'XRP': 'XRPUSDT',      # Ripple - Payment protocol
    'DOGE': 'DOGEUSDT',    # Dogecoin - Community favorite
    'ADA': 'ADAUSDT',      # Cardano - Sustainable blockchain
    'MATIC': 'MATICUSDT',  # Polygon - Ethereum scaling
    'VET': 'VETUSDT',      # VeChain - Supply chain leader
    'HBAR': 'HBARUSDT',    # Hedera - Enterprise blockchain
    'ALGO': 'ALGOUSDT',    # Algorand - Fast & scalable
    'ATOM': 'ATOMUSDT',    # Cosmos - Interoperability hub
    'NEAR': 'NEARUSDT',    # Near Protocol - Developer friendly
    'ARB': 'ARBUSDT'       # Arbitrum - Layer-2 leader
}


def load_dataset(symbol: str, data_dir: str = './data/raw') -> pd.DataFrame:
    """Load preprocessed dataset for a coin"""
    
    # Find the CSV file for this symbol
    pattern = f"{data_dir}/{symbol}_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        logger.error(f"No dataset found for {symbol} in {data_dir}")
        return None
    
    # Use most recent file
    filepath = sorted(files)[-1]
    logger.info(f"Loading: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df):,} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 30+ technical indicators"""
    
    logger.info("Calculating 30+ technical indicators...")
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['returns_2h'] = df['close'].pct_change(2)
    df['returns_4h'] = df['close'].pct_change(4)
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Price relationships
    df['close_sma20_ratio'] = df['close'] / df['sma_20']
    df['sma10_sma50_ratio'] = df['sma_10'] / df['sma_50']
    
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
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-6)
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-6)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Price momentum
    df['momentum_10'] = df['close'].diff(10) / df['close'].shift(10)
    df['momentum_20'] = df['close'].diff(20) / df['close'].shift(20)
    
    # Price position
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    
    logger.info(f"Calculated {len([c for c in df.columns if c.startswith(('returns', 'sma', 'ema', 'rsi', 'bb', 'macd', 'volume', 'volatility', 'momentum', 'price'))])} indicators")
    
    return df


def create_xgboost_labels(df: pd.DataFrame, future_periods: int = 6, threshold: float = 0.01) -> pd.DataFrame:
    """Create classification labels"""
    
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    positive_ratio = df['label'].mean()
    logger.info(f"Label distribution: {positive_ratio:.2%} positive, {(1-positive_ratio):.2%} negative")
    
    return df


def prepare_xgboost_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare XGBoost training data"""
    
    feature_columns = [
        'returns', 'returns_2h', 'returns_4h',
        'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
        'close_sma20_ratio', 'sma10_sma50_ratio',
        'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'macd', 'macd_signal', 'macd_histogram',
        'volume_ratio',
        'volatility', 'momentum_10', 'momentum_20', 'price_position'
    ]
    
    df_clean = df.dropna(subset=feature_columns + ['label'])
    
    logger.info(f"Clean samples: {len(df_clean):,} out of {len(df):,}")
    
    X = df_clean[feature_columns].values
    y = df_clean['label'].values
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    return X_train, y_train, X_val, y_val


def prepare_lstm_data(df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare LSTM training data"""
    
    prices = df['close'].values
    
    X, y = [], []
    for i in range(sequence_length, len(prices) - 1):
        X.append(prices[i-sequence_length:i])
        y.append(prices[i+1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_reshaped = X.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    
    logger.info(f"LSTM Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    return X_train, y_train, X_val, y_val


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray):
    """Train XGBoost classifier"""
    
    try:
        import xgboost as xgb
        
        logger.info("Initializing XGBoost Classifier...")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            evals_result=evals_result,
            verbose_eval=10
        )
        
        # Calculate accuracy
        pred = model.predict(dval)
        pred_labels = (pred > 0.5).astype(int)
        accuracy = (pred_labels == y_val).mean()
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        
        return model, {
            'accuracy': float(accuracy),
            'auc': float(evals_result['eval']['auc'][-1]) if evals_result else 0
        }
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        return None, {}


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray):
    """Train LSTM forecaster"""
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        logger.info("Building LSTM model...")
        
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("Training LSTM...")
        
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        val_mae = float(history.history['val_mae'][-1])
        logger.info(f"LSTM Val MAE: {val_mae:.6f}")
        
        return model, {'val_mae': val_mae, 'final_loss': float(history.history['loss'][-1])}
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        return None, {}


def train_single_coin(symbol: str, coin_name: str) -> Dict:
    """Train models for a single coin"""
    
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING {coin_name} ({symbol})")
    logger.info("="*80)
    
    result = {'symbol': symbol, 'coin_name': coin_name}
    
    try:
        # Load data
        df = load_dataset(symbol)
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}")
            result['status'] = 'failed'
            return result
        
        # Process data
        df = calculate_technical_indicators(df)
        df = create_xgboost_labels(df)
        
        # Train XGBoost
        logger.info("Preparing XGBoost data...")
        X_train, y_train, X_val, y_val = prepare_xgboost_data(df)
        xgb_model, xgb_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
        result['xgboost'] = {'metrics': xgb_metrics}
        
        # Train LSTM
        logger.info("Preparing LSTM data...")
        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm = prepare_lstm_data(df)
        lstm_model, lstm_metrics = train_lstm_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
        result['lstm'] = {'metrics': lstm_metrics}
        
        result['status'] = 'success'
        logger.info(f"[OK] Training complete for {coin_name}")
        
    except Exception as e:
        logger.error(f"Error training {symbol}: {e}", exc_info=True)
        result['status'] = 'failed'
        result['error'] = str(e)
    
    return result


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("= CryptoVolt - Standalone Multi-Coin Model Training")
    print(f"= Training {len(TOP_10_COINS)} coins with large datasets")
    print("="*80 + "\n")
    
    training_results = []
    successful = 0
    
    for idx, (coin_name, symbol) in enumerate(TOP_10_COINS.items(), 1):
        logger.info(f"\n[{idx}/{len(TOP_10_COINS)}]")
        
        result = train_single_coin(symbol, coin_name)
        training_results.append(result)
        
        if result['status'] == 'success':
            successful += 1
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for r in training_results:
        status = "OK" if r['status'] == 'success' else "FAIL"
        logger.info(f"[{status}] {r['coin_name']:8} ({r['symbol']:12})")
        if r['status'] == 'success':
            xgb_acc = r.get('xgboost', {}).get('metrics', {}).get('accuracy', 0)
            lstm_mae = r.get('lstm', {}).get('metrics', {}).get('val_mae', 0)
            logger.info(f"     XGBoost Accuracy: {xgb_acc:.4f}")
            logger.info(f"     LSTM MAE: {lstm_mae:.6f}")
    
    logger.info(f"\nTotal Successful: {successful}/{len(TOP_10_COINS)}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print(f"= Training Complete! {successful}/{len(TOP_10_COINS)} coins trained successfully")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
