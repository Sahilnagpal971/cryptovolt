"""
Quick start script for CryptoVolt large dataset & multi-coin model training
Verifies setup and provides easy commands to run the full pipeline
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n" + "="*70)
    print("Checking Dependencies...")
    print("="*70)
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'requests': 'HTTP requests',
        'xgboost': 'XGBoost models',
        'sklearn': 'Scikit-learn utilities',
        'tensorflow': 'LSTM models',
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package:15} - {description}")
        except ImportError:
            print(f"❌ {package:15} - {description} [MISSING]")
            missing_packages.append(package)
    
    return missing_packages


def check_directories():
    """Check if required directories exist"""
    print("\n" + "="*70)
    print("Checking Directories...")
    print("="*70)
    
    required_dirs = [
        './data',
        './data/raw',
        './models',
        './backend/app',
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"📁 {dir_path} [Will be created]")
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def check_binance_connection():
    """Test Binance API connectivity"""
    print("\n" + "="*70)
    print("Checking Binance API Connection...")
    print("="*70)
    
    try:
        import requests
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
        if response.status_code == 200:
            print("✅ Binance API is reachable")
            return True
        else:
            print(f"⚠️  Binance API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Binance API: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*70)
    print("Setup Instructions")
    print("="*70)
    
    instructions = """
STEP 1: Verify Dependencies
─────────────────────────────
Make sure all required packages are installed:

  pip install pandas numpy requests xgboost scikit-learn tensorflow

STEP 2: Fetch Large Datasets (15-30 minutes)
──────────────────────────────────────────────
Downloads 2 years of hourly data for all 10 coins (~175,000 data points):

  cd backend
  python fetch_large_dataset.py

This will:
  • Fetch BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT,
    AVAXUSDT, MATICUSDT, LINKUSDT, UNIUSDT, DOTUSDT
  • Download ~17,500 hourly candles per coin
  • Save to ./data/raw/SYMBOL_dates.csv
  • Generate dataset_fetch.log with progress

STEP 3: Train Multi-Coin Models (2-3 hours)
────────────────────────────────────────────
Trains XGBoost + LSTM models for each of the 10 coins:

  python train_multi_coin_models.py

This will:
  • Load datasets for all 10 coins
  • Calculate 30+ technical indicators
  • Train XGBoost classification models (1% prediction)
  • Train LSTM forecasting models (60-timestep)
  • Save models to ./models/
  • Generate model_training.log with metrics

STEP 4: Use Trained Models
──────────────────────────
Models are automatically registered and ready to use:

  from app.ml.models import ModelRegistry
  registry = ModelRegistry("./models")
  
  # Load a trained model
  xgb_model = registry.get_model("xgboost_btcusdt", version="v2.0_large")
  lstm_model = registry.get_model("lstm_ethusdt", version="v2.0_large")
  
  # Make predictions
  prediction = xgb_model.predict(features)

TIPS:
────
• Monitor training with: tail -f model_training.log
• Expected XGBoost accuracy: 52-58%
• Expected LSTM MAE: 0.005-0.020
• Enable GPU in train_multi_coin_models.py for faster training
• To use only 365 days of data: modify fetch_large_dataset.py line ~270

IMPORTANT NOTES:
────────────────
✓ Dataset fetch is safe to restart - it's idempotent
✓ Model training will overwrite old models
✓ All data is saved locally
✓ Binance API has request limits - script handles rate limiting

For more details, see: LARGE_DATASET_TRAINING_GUIDE.md
"""
    
    print(instructions)


def print_summary(missing_packages, binance_ok):
    """Print summary and next steps"""
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if missing_packages:
        print(f"⚠️  Missing {len(missing_packages)} package(s):")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("✅ All dependencies are installed!")
    
    if not binance_ok:
        print("\n⚠️  Cannot reach Binance API - check your internet connection")
    else:
        print("✅ Binance API is accessible")
    
    print("\n" + "="*70)
    print("Ready to Start?")
    print("="*70)
    
    if not missing_packages and binance_ok:
        print("\n✅ Your system is ready! Follow the instructions above to:")
        print("   1. Fetch large datasets")
        print("   2. Train multi-coin models")
        print("   3. Start using them in your trading system")
    else:
        print("\n⚠️  Please fix the issues above before starting the training pipeline")


def main():
    """Run all checks and print instructions"""
    
    print("\n" + "█"*70)
    print("█ CryptoVolt Large Dataset & Multi-Coin Training - Quick Start")
    print("█"*70)
    
    missing_packages = check_dependencies()
    check_directories()
    binance_ok = check_binance_connection()
    
    print_setup_instructions()
    print_summary(missing_packages, binance_ok)
    
    print("\n")


if __name__ == "__main__":
    main()
