"""
Fetch large historical datasets for multiple cryptocurrencies from Binance
This script fetches millions of OHLCV data points for top 10 reliable crypto coins
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import time
import os
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top 10 reliable, cheaper, and future-growing coins
TOP_10_COINS = {
    'BTC': 'BTCUSDT',      # Bitcoin - Market leader
    'ETH': 'ETHUSDT',      # Ethereum - Smart contracts
    'SOL': 'SOLUSDT',      # Solana - High-speed
    'XRP': 'XRPUSDT',      # Ripple - Payment protocol
    'ADA': 'ADAUSDT',      # Cardano - PoS blockchain
    'AVAX': 'AVAXUSDT',    # Avalanche - Scalable
    'MATIC': 'MATICUSDT',  # Polygon - Layer-2
    'LINK': 'LINKUSDT',    # Chainlink - Oracles
    'UNI': 'UNIUSDT',      # Uniswap - DeFi
    'DOT': 'DOTUSDT'       # Polkadot - Multi-chain
}

class BinanceDataFetcher:
    """Fetch large historical datasets from Binance"""
    
    def __init__(self, interval: str = '1h', max_retries: int = 3):
        """
        Initialize fetcher
        
        Args:
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            max_retries: Max API call retries
        """
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.interval = interval
        self.max_retries = max_retries
        self.request_count = 0
        self.rate_limit_reset = None
        
    def _make_request(self, params: dict) -> list:
        """Make API request with retry logic and rate limit handling"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                self.request_count += 1
                
                if self.request_count % 50 == 0:
                    logger.info(f"Completed {self.request_count} API requests")
                
                return response.json()
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    raise
        
        return []
    
    def fetch_historical_data(self, symbol: str, days_back: int = 365) -> pd.DataFrame:
        """
        Fetch large historical dataset for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            days_back: Number of days to fetch (default 365 = ~8760 hourly candles)
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching {days_back} days of {self.interval} data for {symbol}")
        logger.info(f"{'='*60}")
        
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Calculate interval in milliseconds
        interval_ms = self._get_interval_ms(self.interval)
        total_candles_needed = (days_back * 24 * 60 * 60 * 1000) // interval_ms
        
        logger.info(f"Total candles to fetch: ~{total_candles_needed:,}")
        
        batch_size = 1000  # Binance limit per request
        batches = (total_candles_needed + batch_size - 1) // batch_size
        
        start_time = end_time - (interval_ms * total_candles_needed)
        
        logger.info(f"Fetching in {batches} batches of {batch_size} candles...")
        
        for batch_num in range(batches):
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": int(start_time),
                "endTime": int(end_time),
                "limit": batch_size
            }
            
            data = self._make_request(params)
            
            if not data:
                logger.warning(f"No data returned for {symbol}, breaking loop")
                break
            
            all_data.extend(data)
            
            # Update end_time for next batch (go backwards in time)
            if len(data) > 0:
                last_timestamp = int(data[-1][0])
                end_time = last_timestamp - interval_ms
                
                progress = min(len(all_data), total_candles_needed)
                logger.info(f"Batch {batch_num + 1}/{batches}: Fetched {len(data)} candles (Total: {progress:,}/{total_candles_needed:,})")
                
                # Small delay between requests to avoid rate limiting
                if batch_num < batches - 1:
                    time.sleep(0.1)
            else:
                logger.warning(f"Empty batch returned for {symbol}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to appropriate types
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # Remove NaN rows
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        logger.info(f"\n✅ Successfully fetched {len(df):,} unique candles for {symbol}")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"   Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        return df
    
    def _get_interval_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return intervals.get(interval, 60 * 60 * 1000)


def save_dataset(df: pd.DataFrame, symbol: str, output_dir: str = './data/raw'):
    """Save dataset to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/{symbol}_{df['timestamp'].min().date()}_{df['timestamp'].max().date()}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved to: {filename}")
    logger.info(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    return filename


def fetch_all_coins_data(days_back: int = 365, interval: str = '1h') -> dict:
    """Fetch data for all top 10 coins"""
    
    logger.info("\n" + "="*80)
    logger.info("CRYPTOVOLT - LARGE DATASET FETCHING")
    logger.info(f"Fetching {days_back} days of historical data for top 10 coins")
    logger.info(f"Interval: {interval}")
    logger.info("="*80)
    
    fetcher = BinanceDataFetcher(interval=interval)
    datasets = {}
    data_files = {}
    
    for coin_name, symbol in TOP_10_COINS.items():
        try:
            df = fetcher.fetch_historical_data(symbol, days_back=days_back)
            datasets[symbol] = df
            
            filepath = save_dataset(df, symbol)
            data_files[symbol] = filepath
            
            # Small delay between coin fetches
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FETCH SUMMARY")
    logger.info("="*80)
    
    total_candles = sum(len(df) for df in datasets.values())
    
    for symbol, df in datasets.items():
        coin_name = [k for k, v in TOP_10_COINS.items() if v == symbol][0]
        logger.info(f"{coin_name:8} ({symbol:12}): {len(df):8,} candles | {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    logger.info(f"\nTotal candles across all coins: {total_candles:,}")
    logger.info(f"Total Binance API requests: {fetcher.request_count}")
    
    return datasets, data_files


if __name__ == "__main__":
    # Fetch 2 years of hourly data for each coin (approximately 17,500 candles per coin)
    # This gives us a large, recent dataset for robust model training
    datasets, data_files = fetch_all_coins_data(days_back=730, interval='1h')
    
    logger.info("\n✅ Dataset fetching complete!")
    logger.info(f"Data saved to ./data/raw/")
    logger.info(f"Ready for model training with {sum(len(df) for df in datasets.values()):,} total data points")
