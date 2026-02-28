"""
Enhanced multi-coin model training pipeline for CryptoVolt
Trains XGBoost and LSTM models for each of the top 10 coins
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import pickle
import json
from typing import Dict, Tuple, List
from app.ml.models import XGBoostClassifier, LSTMForecaster, ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top 10 cheap, reliable, and future-growing coins
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
    
    import glob
    
    # Find the CSV file for this symbol
    pattern = f"{data_dir}/{symbol}_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        logger.error(f"No dataset found for {symbol} in {data_dir}")
        return None
    
    # Use most recent file
    filepath = sorted(files)[-1]
    logger.info(f"Loading dataset: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df):,} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    
    logger.info("Calculating technical indicators...")
    
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
    
    # RSI zones
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_trend'] = df['volume'].rolling(window=10).mean()
    
    # Volatility measures
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_ema'] = df['returns'].ewm(span=20, adjust=False).std()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Price momentum
    df['momentum_10'] = df['close'].diff(10) / df['close'].shift(10)
    df['momentum_20'] = df['close'].diff(20) / df['close'].shift(20)
    
    # Price position in daily range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Stochastic oscillator
    min_price = df['low'].rolling(window=14).min()
    max_price = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - min_price) / (max_price - min_price)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    logger.info(f"Calculated {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore', 'close_time']])} technical indicators")
    
    return df


def create_xgboost_labels(df: pd.DataFrame, future_periods: int = 6, threshold: float = 0.01) -> pd.DataFrame:
    """Create classification labels for XGBoost"""
    
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    # Calculate class balance
    positive_ratio = df['label'].mean()
    logger.info(f"Label distribution: {positive_ratio:.2%} positive, {(1-positive_ratio):.2%} negative")
    
    return df


def prepare_xgboost_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature matrix for XGBoost"""
    
    feature_columns = [
        'returns', 'returns_2h', 'returns_4h',
        'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
        'close_sma20_ratio', 'sma10_sma50_ratio',
        'rsi', 'rsi_overbought', 'rsi_oversold',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'macd', 'macd_signal', 'macd_histogram',
        'volume_ratio', 'volume_trend',
        'volatility', 'volatility_ema', 'high_low_ratio',
        'momentum_10', 'momentum_20',
        'price_position', 'stoch_k', 'stoch_d', 'atr'
    ]
    
    # Drop NaN rows
    df_clean = df.dropna(subset=feature_columns + ['label'])
    
    logger.info(f"Using {len(df_clean):,} clean samples out of {len(df):,} total")
    
    X = df_clean[feature_columns].values
    y = df_clean['label'].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/val (80/20)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Data split: Train={len(X_train):,} (80%), Val={len(X_val):,} (20%)")
    
    return X_train, y_train, X_val, y_val


def prepare_lstm_data(df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare sequences for LSTM"""
    
    # Use close prices normalized by their history
    prices = df['close'].values
    
    X, y = [], []
    for i in range(sequence_length, len(prices) - 1):
        X.append(prices[i-sequence_length:i])
        y.append(prices[i+1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize using MinMaxScaler for prices
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_reshaped = X.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    
    # Reshape for LSTM [samples, timesteps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    # Normalize y
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    # Split (80/20)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    
    logger.info(f"LSTM data: Train={len(X_train):,}, Val={len(X_val):,}")
    
    return X_train, y_train, X_val, y_val


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """Train XGBoost classifier"""
    
    logger.info("="*70)
    logger.info("Training XGBoost Classifier")
    logger.info("="*70)
    
    model = XGBoostClassifier(params={
        "max_depth": 7,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # More efficient
        "device": "cuda" if False else None  # Set to cuda if available
    })
    
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=150)
    
    logger.info(f"XGBoost Training Complete")
    logger.info(f"  Metrics: {json.dumps(metrics, indent=2)}")
    
    return model, metrics


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """Train LSTM forecaster"""
    
    logger.info("="*70)
    logger.info("Training LSTM Forecaster")
    logger.info("="*70)
    
    model = LSTMForecaster(sequence_length=60)
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    logger.info(f"LSTM Training Complete")
    logger.info(f"  Metrics: {json.dumps(metrics, indent=2)}")
    
    return model, metrics


def train_coin_models(symbol: str, coin_name: str, data_dir: str = './data/raw') -> Dict:
    """Train both XGBoost and LSTM models for a specific coin"""
    
    logger.info("\n" + "█"*80)
    logger.info(f"█ TRAINING MODELS FOR {coin_name} ({symbol})")
    logger.info("█"*80)
    
    results = {'symbol': symbol, 'coin_name': coin_name}
    
    try:
        # Load dataset
        df = load_dataset(symbol, data_dir)
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}")
            results['status'] = 'failed'
            return results
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        df = create_xgboost_labels(df, future_periods=6, threshold=0.01)
        
        # Train XGBoost
        logger.info(f"\n--- XGBoost Training for {coin_name} ---")
        X_train, y_train, X_val, y_val = prepare_xgboost_data(df)
        xgb_model, xgb_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
        results['xgboost'] = {
            'model': xgb_model.model,
            'metrics': xgb_metrics,
            'config': xgb_model.params
        }
        
        # Train LSTM
        logger.info(f"\n--- LSTM Training for {coin_name} ---")
        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm = prepare_lstm_data(df)
        lstm_model, lstm_metrics = train_lstm_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
        results['lstm'] = {
            'model': lstm_model.model,
            'metrics': lstm_metrics,
            'config': {'sequence_length': 60}
        }
        
        results['status'] = 'success'
        logger.info(f"\n✅ Training complete for {coin_name}")
        
    except Exception as e:
        logger.error(f"❌ Error training models for {symbol}: {e}", exc_info=True)
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def main():
    """Main multi-coin training pipeline"""
    
    print("\n" + "="*80)
    print("CryptoVolt - Multi-Coin ML Model Training Pipeline")
    print(f"Training {len(TOP_10_COINS)} coins with large datasets (~730 days of hourly data)")
    print("="*80 + "\n")
    
    # Initialize model registry
    registry = ModelRegistry("./models")
    
    training_results = []
    successful_coins = 0
    
    for coin_idx, (coin_name, symbol) in enumerate(TOP_10_COINS.items(), 1):
        logger.info(f"\n[{coin_idx}/{len(TOP_10_COINS)}]")
        
        result = train_coin_models(symbol, coin_name)
        training_results.append(result)
        
        if result['status'] == 'success':
            successful_coins += 1
            
            # Register models in registry
            try:
                xgb_model = result['xgboost']['model']
                lstm_model = result['lstm']['model']
                
                xgb_id = registry.register_model(
                    model=xgb_model,
                    model_name=f"xgboost_{symbol.lower()}",
                    model_type="xgboost",
                    version="v2.0_large",
                    metrics=result['xgboost']['metrics'],
                    config=result['xgboost']['config']
                )
                
                lstm_id = registry.register_model(
                    model=lstm_model,
                    model_name=f"lstm_{symbol.lower()}",
                    model_type="lstm",
                    version="v2.0_large",
                    metrics=result['lstm']['metrics'],
                    config=result['lstm']['config']
                )
                
                logger.info(f"✅ Registered XGBoost: {xgb_id}")
                logger.info(f"✅ Registered LSTM: {lstm_id}")
                
            except Exception as e:
                logger.error(f"Failed to register models: {e}")
    
    # Summary Report
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for result in training_results:
        status_symbol = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_symbol} {result['coin_name']:8} ({result['symbol']:12}): {result['status']}")
        
        if result['status'] == 'success':
            xgb_acc = result['xgboost']['metrics'].get('accuracy', 0)
            lstm_mae = result['lstm']['metrics'].get('val_mae', result['lstm']['metrics'].get('final_mae', 0))
            print(f"   - XGBoost Accuracy: {xgb_acc:.4f}")
            print(f"   - LSTM Val MAE: {lstm_mae:.6f}")
    
    print(f"\nTotal Successful: {successful_coins}/{len(TOP_10_COINS)}")
    print(f"Models Location: ./models/")
    print("\n" + "="*80)
    print("✅ Multi-Coin Training Pipeline Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
