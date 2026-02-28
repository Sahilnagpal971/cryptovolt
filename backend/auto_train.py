"""
Auto-training script: Waits for dataset fetching to complete, then trains models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import glob
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected coins (10 total)
EXPECTED_COINS = [
    'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'MATICUSDT', 'VETUSDT',
    'HBARUSDT', 'ALGOUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT'
]

def count_dataset_files(data_dir='./data/raw'):
    """Count how many dataset files have been downloaded"""
    files = glob.glob(f"{data_dir}/*.csv")
    return len(files)

def wait_for_datasets(max_wait_minutes=45, check_interval=10):
    """
    Wait for dataset fetching to complete
    
    Args:
        max_wait_minutes: Maximum time to wait for datasets
        check_interval: Time between checks in seconds
    """
    logger.info("="*80)
    logger.info("MONITORING DATASET FETCHING")
    logger.info("="*80)
    logger.info(f"Waiting for {len(EXPECTED_COINS)} coin datasets to be downloaded...")
    logger.info(f"Check interval: {check_interval} seconds")
    logger.info(f"Max wait time: {max_wait_minutes} minutes\n")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_count = 0
    
    while True:
        elapsed = time.time() - start_time
        elapsed_minutes = elapsed / 60
        
        current_count = count_dataset_files()
        
        if current_count > last_count:
            logger.info(f"[{elapsed_minutes:.1f}m] Downloaded {current_count} dataset files")
            last_count = current_count
        
        # Check if all datasets are fetched
        if current_count >= len(EXPECTED_COINS):
            logger.info("\n" + "="*80)
            logger.info(f"[OK] All {current_count} dataset files downloaded!")
            logger.info("="*80 + "\n")
            return True
        
        # Check for timeout
        if elapsed > max_wait_seconds:
            logger.error(f"Timeout reached after {max_wait_minutes} minutes")
            logger.error(f"Only {current_count}/{len(EXPECTED_COINS)} datasets fetched")
            return False
        
        # Wait before next check
        time.sleep(check_interval)

def start_training():
    """Start the multi-coin model training"""
    logger.info("="*80)
    logger.info("STARTING MULTI-COIN MODEL TRAINING")
    logger.info("="*80 + "\n")
    
    import subprocess
    
    # Run training script
    try:
        result = subprocess.run(
            [sys.executable, 'train_multi_coin_models.py'],
            cwd=Path(__file__).parent,
            check=False
        )
        
        if result.returncode == 0:
            logger.info("\n" + "="*80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            return True
        else:
            logger.error(f"Training failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return False

def main():
    """Main workflow: Wait for datasets, then train"""
    
    print("\n" + "█"*80)
    print("█ CryptoVolt - Auto Dataset Fetch & Model Training Pipeline")
    print("█"*80 + "\n")
    
    # Step 1: Wait for datasets
    logger.info("\nSTEP 1: Wait for dataset fetching to complete...")
    datasets_ready = wait_for_datasets(max_wait_minutes=45, check_interval=15)
    
    if not datasets_ready:
        logger.error("Dataset fetching did not complete in time")
        return False
    
    # Step 2: Verify datasets exist
    logger.info("\nSTEP 2: Verifying downloaded datasets...")
    data_dir = './data/raw'
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    files = glob.glob(f"{data_dir}/*.csv")
    logger.info(f"Found {len(files)} dataset files:")
    for f in sorted(files):
        size_mb = os.path.getsize(f) / (1024*1024)
        logger.info(f"  - {os.path.basename(f)} ({size_mb:.2f} MB)")
    
    # Step 3: Start training
    logger.info("\nSTEP 3: Starting model training...")
    training_success = start_training()
    
    return training_success

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "█"*80)
        print("█ PIPELINE COMPLETE! All datasets fetched and models trained.")
        print("█"*80 + "\n")
    else:
        print("\n" + "█"*80)
        print("█ PIPELINE FAILED! Check logs for details.")
        print("█"*80 + "\n")
