"""
Improved large-scale dataset fetcher for CryptoVolt
Fetches complete 2-year historical data more reliably
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import requests
import time
import os
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top 10 cheap, growth-focused coins
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


class ImprovedBinanceFetcher:
    """Improved Binance data fetcher with better error handling"""
    
    def __init__(self, interval='1d'):  # Using daily bars instead of hourly for more data
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.interval = interval
        self.session = requests.Session()
        self.request_count = 0
        
    def fetch_all_historical_data(self, symbol: str, days_back: int = 730) -> pd.DataFrame:
        """
        Fetch complete historical data more reliably
        Using daily candles (1d) instead of hourly to get more historical data
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Fetching {days_back} days of {self.interval} data for {symbol}")
        logger.info(f"{'='*70}")
        
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Calculate time back
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        logger.info(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.now()}")
        logger.info(f"Fetching in batches of 1000...")
        
        current_time = start_time
        batch_num = 0
        
        while current_time < end_time:
            batch_num += 1
            
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": int(current_time),
                "endTime": int(end_time),
                "limit": 1000
            }
            
            try:
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.info(f"No more data available. Stopping.")
                    break
                
                all_data.extend(data)
                
                # Update time for next batch (move backwards)
                last_timestamp = int(data[-1][0])
                current_time = last_timestamp + 1
                
                progress = len(all_data)
                logger.info(f"Batch {batch_num}: Fetched {len(data)} candles (Total: {progress:,})")
                
                # Rate limiting
                time.sleep(0.1)
                self.request_count += 1
                
                if len(data) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                time.sleep(5)
                continue
        
        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric and datetime
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        logger.info(f"[OK] Successfully fetched {len(df):,} unique candles")
        logger.info(f"     From: {df['timestamp'].min()}")
        logger.info(f"     To:   {df['timestamp'].max()}")
        logger.info(f"     Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, symbol: str, output_dir='./data/raw'):
        """Save dataset to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{symbol}_improved_{df['timestamp'].min().date()}_{df['timestamp'].max().date()}.csv"
        df.to_csv(filename, index=False)
        
        size_mb = os.path.getsize(filename) / (1024*1024)
        logger.info(f"Saved: {filename}")
        logger.info(f"Size: {size_mb:.2f} MB")
        
        return filename


def main():
    """Main improved data fetching"""
    
    print("\n" + "="*80)
    print("CryptoVolt - Improved Large-Scale Data Fetching")
    print(f"Fetching 2 years of DAILY data for top 10 coins")
    print("="*80 + "\n")
    
    logger.info("Starting improved data fetch process...")
    logger.info(f"Using interval: 1d (daily candles)")
    logger.info(f"Expected data points per coin: ~730")
    
    fetcher = ImprovedBinanceFetcher(interval='1d')  # Daily data
    
    datasets_info = {}
    successful = 0
    
    for coin_name, symbol in TOP_10_COINS.items():
        try:
            df = fetcher.fetch_all_historical_data(symbol, days_back=730)
            
            if len(df) > 100:
                filepath = fetcher.save_dataset(df, symbol)
                datasets_info[symbol] = {
                    'rows': len(df),
                    'path': filepath,
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                }
                successful += 1
                time.sleep(1)
            else:
                logger.warning(f"Skipped {symbol}: insufficient data ({len(df)} rows)")
                
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FETCH SUMMARY")
    logger.info("="*80)
    
    total_rows = sum(info['rows'] for info in datasets_info.values())
    
    for symbol, info in sorted(datasets_info.items()):
        logger.info(f"{symbol:12} - {info['rows']:6,} rows ({info['start'].date()} to {info['end'].date()})")
    
    logger.info(f"\nTotal data points: {total_rows:,}")
    logger.info(f"Successful coins: {successful}/{len(TOP_10_COINS)}")
    logger.info(f"API requests made: {fetcher.request_count}")
    logger.info("="*80)
    
    print(f"\n[OK] Data fetching complete!")
    print(f"     Fetched: {successful}/{len(TOP_10_COINS)} coins")
    print(f"     Total data points: {total_rows:,}")
    print(f"     Files saved to: ./data/raw/")


if __name__ == "__main__":
    main()
