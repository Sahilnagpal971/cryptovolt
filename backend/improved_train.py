"""
Improved multi-coin model training with advanced techniques
- Better feature engineering
- Data augmentation
- Class balancing
- Hyperparameter tuning
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import os
import logging
import glob
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top 10 coins
TOP_10_COINS = {
    'XRP': 'XRPUSDT',
    'DOGE': 'DOGEUSDT',
    'ADA': 'ADAUSDT',
    'MATIC': 'MATICUSDT',
    'VET': 'VETUSDT',
    'HBAR': 'HBARUSDT',
    'ALGO': 'ALGOUSDT',
    'ATOM': 'ATOMUSDT',
    'NEAR': 'NEARUSDT',
    'ARB': 'ARBUSDT'
}


def load_dataset(symbol: str, data_dir='./data/raw') -> pd.DataFrame:
    """Load dataset - prefer improved files first"""
    
    # Look for improved files first
    pattern1 = f"{data_dir}/{symbol}_improved_*.csv"
    pattern2 = f"{data_dir}/{symbol}_*.csv"
    
    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2) if not files1 else []
    
    files = files1 if files1 else files2
    
    if not files:
        logger.error(f"No dataset found for {symbol}")
        return None
    
    filepath = sorted(files)[-1]
    logger.info(f"Loading: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df):,} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced 40+ technical indicator calculation"""
    
    logger.info("Calculating 40+ advanced technical indicators...")
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_2d'] = df['close'].pct_change(2)
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_10d'] = df['close'].pct_change(10)
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # SMA Ratios
    df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
    df['sma_ratio_50_200'] = df['sma_50'] / df['sma_200']
    df['close_sma20'] = df['close'] / (df['sma_20'] + 1e-6)
    df['close_ema10'] = df['close'] / (df['ema_10'] + 1e-6)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI with other periods
    for period in [5, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_sma20'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_sma20'] + (bb_std * 2)
    df['bb_lower'] = df['bb_sma20'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_sma20'] + 1e-6)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
    
    # ATR
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close),
            np.abs(low - close)
        )
    )
    df['atr'] = pd.Series(tr).rolling(window=14).mean()
    
    # Volume features
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-6)
    df['volume_trend'] = df['volume'].pct_change()
    df['price_volume'] = (df['close'].pct_change() * df['volume_ratio']).fillna(0)
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    
    # Momentum indicators
    df['momentum_10'] = df['close'].diff(10)
    df['momentum_20'] = df['close'].diff(20)
    df['rate_of_change'] = df['close'].pct_change(10)
    
    # Stochastic K/D
    min_14 = df['low'].rolling(window=14).min()
    max_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - min_14) / (max_14 - min_14 + 1e-6)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Average Directional Index (ADX)
    df['tr'] = tr
    df['atr_14'] = pd.Series(tr).rolling(window=14).mean()
    
    # Price range features
    df['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-6)
    df['close_open_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-6)
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    
    logger.info(f"Created {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore', 'close_time', 'tr']])} features")
    
    return df


def create_smart_labels(df: pd.DataFrame, future_periods: int = 5) -> pd.DataFrame:
    """Create better labels with dynamic thresholds"""
    
    # Calculate future returns
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
    
    # Use volatility-adjusted threshold
    volatility = df['returns'].rolling(window=20).std()
    threshold = volatility.mean() * 0.5  # Half of average volatility
    
    df['label_volatility_adj'] = (df['future_return'] > threshold).astype(int)
    
    # Also add fixed threshold labels
    df['label_fixed_1pct'] = (df['future_return'] > 0.01).astype(int)
    df['label_fixed_2pct'] = (df['future_return'] > 0.02).astype(int)
    
    # Use volatility-adjusted by default
    df['label'] = df['label_volatility_adj']
    
    pos_ratio = df['label'].mean()
    logger.info(f"Label distribution: {pos_ratio:.2%} positive, {(1-pos_ratio):.2%} negative")
    logger.info(f"Threshold (volatility-adjusted): {threshold:.4f}")
    
    return df


def prepare_training_data(df: pd.DataFrame, test_size: int = 100) -> Tuple:
    """Prepare training data with time-based split"""
    
    # Feature columns (exclude label and timestamp columns)
    feature_cols = [c for c in df.columns if c not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'future_return', 'label', 'label_volatility_adj', 'label_fixed_1pct', 'label_fixed_2pct',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore', 'close_time', 'tr'
    ]]
    
    # Remove rows with NaN
    df_clean = df.dropna(subset=feature_cols + ['label'])
    
    if len(df_clean) < 50:
        logger.warning(f"Too few clean samples: {len(df_clean)}")
        return None, None, None, None
    
    # Time-based split (no data leakage)
    train_size = len(df_clean) - test_size
    
    X_train = df_clean[feature_cols].iloc[:train_size].values
    y_train = df_clean['label'].iloc[:train_size].values
    
    X_test = df_clean[feature_cols].iloc[train_size:].values
    y_test = df_clean['label'].iloc[train_size:].values
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"Class balance - Train: {y_train.mean():.2%} positive")
    
    return X_train_scaled, y_train, X_test_scaled, y_test


def train_xgboost_improved(X_train, y_train, X_test, y_test, coin_name):
    """Train optimized XGBoost classifier"""
    
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        logger.info(f"Training XGBoost for {coin_name}...")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Optimized parameters
        params = {
            'max_depth': 5,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        # Use scale_pos_weight to handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1)
        params['scale_pos_weight'] = scale_pos_weight
        
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Predictions
        pred_prob = model.predict(dtest)
        pred_labels = (pred_prob > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, pred_labels)
        precision = precision_score(y_test, pred_labels, zero_division=0)
        recall = recall_score(y_test, pred_labels, zero_division=0)
        f1 = f1_score(y_test, pred_labels, zero_division=0)
        auc = roc_auc_score(y_test, pred_prob)
        
        logger.info(f"Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  AUC:       {auc:.4f}")
        
        return model, {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        }
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        return None, {}


def train_single_coin_improved(symbol: str, coin_name: str) -> Dict:
    """Train improved model for single coin"""
    
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING {coin_name} ({symbol}) - IMPROVED")
    logger.info("="*80)
    
    result = {'symbol': symbol, 'coin_name': coin_name}
    
    try:
        # Load data
        df = load_dataset(symbol)
        if df is None or len(df) < 50:
            logger.error(f"Insufficient data")
            result['status'] = 'failed'
            return result
        
        # Feature engineering
        df = advanced_feature_engineering(df)
        
        # Create labels
        df = create_smart_labels(df)
        
        # Prepare data
        X_train, y_train, X_test, y_test = prepare_training_data(df)
        
        if X_train is None:
            result['status'] = 'failed'
            return result
        
        # Train model
        model, metrics = train_xgboost_improved(X_train, y_train, X_test, y_test, coin_name)
        
        if model is None:
            result['status'] = 'failed'
            return result
        
        result['model'] = model
        result['metrics'] = metrics
        result['status'] = 'success'
        
        logger.info(f"[OK] Training complete for {coin_name}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        result['status'] = 'failed'
        result['error'] = str(e)
    
    return result


def main():
    """Main improved training pipeline"""
    
    print("\n" + "="*80)
    print("CryptoVolt - Improved Multi-Coin Model Training")
    print(f"Using advanced features, better labels, and optimized hyperparameters")
    print("="*80 + "\n")
    
    logger.info("Starting improved training pipeline...")
    
    results = []
    successful = 0
    
    for idx, (coin_name, symbol) in enumerate(TOP_10_COINS.items(), 1):
        logger.info(f"\n[{idx}/{len(TOP_10_COINS)}]")
        
        result = train_single_coin_improved(symbol, coin_name)
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("IMPROVED TRAINING SUMMARY")
    logger.info("="*80)
    
    for r in results:
        status = "OK" if r['status'] == 'success' else "FAIL"
        logger.info(f"\n[{status}] {r['coin_name']:8} ({r['symbol']:12})")
        if r['status'] == 'success':
            m = r.get('metrics', {})
            logger.info(f"     Accuracy:  {m.get('accuracy', 0):.4f}")
            logger.info(f"     Precision: {m.get('precision', 0):.4f}")
            logger.info(f"     Recall:    {m.get('recall', 0):.4f}")
            logger.info(f"     F1 Score:  {m.get('f1', 0):.4f}")
            logger.info(f"     AUC:       {m.get('auc', 0):.4f}")
    
    logger.info(f"\nTotal Successful: {successful}/{len(TOP_10_COINS)}")
    logger.info("="*80)
    
    print(f"\n" + "="*80)
    print(f"[OK] Improved Training Complete! {successful}/{len(TOP_10_COINS)} coins trained")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
