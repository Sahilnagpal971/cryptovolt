# CryptoVolt API Integration Guide

## Overview
This document provides complete API integration specifications for CryptoVolt, including authentication, rate limits, endpoints, and error handling.

---

## 1. BINANCE FUTURES API

### Authentication
**Type**: HMAC SHA256 Signature Authentication

```python
# Configuration
BINANCE_API_KEY = 'Yo5r7wPiEq4fq1pvU2BSynXJes6MPImKWi03S9rzqbySfu4SJP9mbgdPq0T5YAsT'
BINANCE_API_SECRET = 'lgHUKyJ8J1tJk1PhELNlN32KeNdnHJv940IQ5yJg7OoLdH1l7LJLnSQrpJKF9IU0'
BINANCE_BASE_URL = 'https://fapi.binance.com'  # Futures API
```

### Key Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/fapi/v1/klines` | Get candlestick data |
| GET | `/fapi/v1/ticker/24hr` | 24-hour ticker stats |
| GET | `/fapi/v1/depth` | Get order book depth |
| GET | `/fapi/v1/trades` | Get recent trades |
| POST | `/fapi/v1/order` | Place order (BUY/SELL) |
| GET | `/fapi/v1/openOrders` | Get open orders |
| GET | `/fapi/v1/allOrders` | Get all orders |
| POST | `/fapi/v1/order/test` | Test order (paper trading) |
| GET | `/fapi/v1/account` | Get account information |
| GET | `/fapi/v1/balance` | Get account balance |
| GET | `/fapi/v2/positionRisk` | Get position information |
| DELETE | `/fapi/v1/order` | Cancel order |
| WebSocket | `wss://fstream.binance.com/ws` | Real-time market data stream |

### Rate Limits
- **Requests**: 1200 requests per minute
- **Order Placement**: 100 orders per 10 seconds
- **WebSocket**: Unlimited connections, ~100 messages per second

### Order Types
1. **MARKET** - Execute immediately at market price
2. **LIMIT** - Execute at specific price
3. **STOP_LOSS** - Stop-loss order
4. **TAKE_PROFIT** - Take-profit order

### Position Management
- **Leverage**: 1x to 125x (configurable per symbol)
- **Margin Mode**: Isolated or Cross
- **Default**: Isolated mode for safety

### Example: Place Paper Trading Order
```python
import hmac
import hashlib
import requests
import time

def place_paper_order(symbol, side, quantity, price=None, order_type='MARKET'):
    """
    Place a paper (test) order on Binance Futures
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        price: Price (only for LIMIT orders)
        order_type: 'MARKET' or 'LIMIT'
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/order/test"
    
    params = {
        'symbol': symbol,
        'side': side,
        'type': order_type,
        'quantity': quantity,
        'timestamp': int(time.time() * 1000),
    }
    
    if order_type == 'LIMIT' and price:
        params['price'] = price
        params['timeInForce'] = 'GTC'  # Good Till Cancel
    
    # Generate signature
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        BINANCE_API_SECRET.encode(),
        query_string.encode(),
        hashlib.sha256
    ).hexdigest()
    params['signature'] = signature
    
    headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
    
    try:
        response = requests.post(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            print(f"✅ Paper order test successful: {response.json()}")
            return True
        else:
            print(f"❌ Order test failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
```

### WebSocket Stream Example
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    # Process real-time candlestick or trade data
    print(f"🔔 Market Update: {data}")

def setup_market_stream(symbol='BTCUSDT', interval='1m'):
    """Subscribe to real-time market data"""
    stream_name = f"{symbol.lower()}@kline_{interval}"
    ws_url = f"wss://fstream.binance.com/ws/{stream_name}"
    
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message
    )
    ws.run_forever()
```

---

## 2. DISCORD ALERTS API

### Authentication
**Type**: Webhook Token

```python
# Configuration
DISCORD_BOT_TOKEN = '2wLqR-vGkcFYILQekphBMRoYlbIvuyqy'
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1429173388814844026/16b7qD0GIeCjV0Ul4bwDpSnM4biOz7q56X1mtcF2vQ2QnqOH0wkRv2xAd_HMVZoRNw5O'
DISCORD_CHANNEL_ID = '1429172043202560163'
```

### Webhook Integration
**Endpoint**: `https://discord.com/api/webhooks/{WEBHOOK_ID}/{WEBHOOK_TOKEN}`

**Use Cases**:
- Real-time trade execution alerts
- Sentiment analysis updates
- Risk warnings
- System health notifications

### Message Types

#### 1. Trade Signal Alert
```python
import requests
import json

def send_trade_alert(signal_type, symbol, confidence, price, action):
    """Send trading signal to Discord"""
    
    color_map = {'BUY': 0x00FF00, 'SELL': 0xFF0000, 'HOLD': 0xFFFF00}
    
    embed = {
        "title": f"🎯 Trading Signal: {signal_type}",
        "description": f"Symbol: **{symbol}** | Price: **${price}**",
        "color": color_map.get(action, 0xFFFF00),
        "fields": [
            {
                "name": "Signal Type",
                "value": signal_type,
                "inline": True
            },
            {
                "name": "Confidence",
                "value": f"{confidence:.2%}",
                "inline": True
            },
            {
                "name": "Action",
                "value": action,
                "inline": False
            }
        ],
        "footer": {"text": "CryptoVolt Trading System"}
    }
    
    payload = {"embeds": [embed]}
    
    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.status_code == 204
```

#### 2. Sentiment Analysis Update
```python
def send_sentiment_alert(coin, sentiment, score, confidence):
    """Send sentiment analysis update to Discord"""
    
    emoji_map = {'BULLISH': '📈', 'BEARISH': '📉', 'NEUTRAL': '➡️'}
    color_map = {'BULLISH': 0x00FF00, 'BEARISH': 0xFF0000, 'NEUTRAL': 0x808080}
    
    embed = {
        "title": f"{emoji_map[sentiment]} Market Sentiment: {coin.upper()}",
        "color": color_map[sentiment],
        "fields": [
            {"name": "Sentiment", "value": sentiment, "inline": True},
            {"name": "Score", "value": f"{score:.4f}", "inline": True},
            {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True}
        ]
    }
    
    payload = {"embeds": [embed]}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    
    return response.status_code == 204
```

#### 3. Risk Warning Alert
```python
def send_risk_alert(risk_level, description, positions_at_risk):
    """Send risk warning to Discord"""
    
    color_map = {'HIGH': 0xFF0000, 'MEDIUM': 0xFF8800, 'LOW': 0xFFFF00}
    
    embed = {
        "title": f"⚠️ Risk Alert: {risk_level}",
        "color": color_map[risk_level],
        "description": description,
        "fields": [
            {"name": "Positions at Risk", "value": str(positions_at_risk)},
            {"name": "Recommended Action", "value": "Review immediately"}
        ]
    }
    
    payload = {"embeds": [embed]}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    
    return response.status_code == 204
```

### Rate Limits
- **Webhooks**: 10 messages per 10 seconds per webhook
- **Burst**: 20 messages maximum before rate limiting

---

## 3. REDDIT API (PRAW)

### OAuth Configuration
**Type**: Reddit Application Authentication

```python
REDDIT_CONFIG = {
    'client_id': 'oXvmcPz6Sb2ObD5q9FQ0dw',
    'client_secret': 'RbNtP24dpX_S2t19fbIVTlz-AeZYYA',
    'user_agent': 'CryptoVolt/3.0 by u/Ill-Database-3830'
}
```

### Setup Instructions
1. **Create Reddit App**:
   - Go to: https://www.reddit.com/prefs/apps
   - Click "Create App"
   - Name: `CryptoVolt`
   - Type: `script`
   - Redirect URI: `http://localhost:8080`

2. **Get Credentials**:
   - `client_id`: The string under app name
   - `client_secret`: Copy from the app details

### PRAW Features
- **Read-Only Mode**: No authentication required (but slower)
- **OAuth Mode**: Full authentication (faster, recommended)
- **Rate Limits**: 60 requests per minute

### Example Usage
```python
import praw

def fetch_reddit_posts(subreddit_name, limit=150):
    """Fetch Reddit posts for sentiment analysis"""
    
    reddit = praw.Reddit(
        client_id=REDDIT_CONFIG['client_id'],
        client_secret=REDDIT_CONFIG['client_secret'],
        user_agent=REDDIT_CONFIG['user_agent']
    )
    
    subreddit = reddit.subreddit(subreddit_name)
    
    posts = []
    for submission in subreddit.hot(limit=limit):
        posts.append({
            'title': submission.title,
            'score': submission.score,
            'comments': submission.num_comments,
            'created': submission.created_utc,
            'author_karma': submission.author.comment_karma if submission.author else 0
        })
    
    return posts
```

### Key Endpoints (via PRAW)
| Method | Purpose |
|--------|---------|
| `subreddit.hot()` | Get hot posts |
| `subreddit.new()` | Get newest posts |
| `subreddit.search()` | Search posts |
| `submission.comments` | Get post comments |
| `redditor.comment_karma` | Get user karma |

---

## 4. NEWS & SENTIMENT SOURCES

### CryptoCompare API
**Endpoint**: `https://min-api.cryptocompare.com/data/v2/news/`
**Method**: GET
**Auth**: Optional API key
**Rate**: 100,000 calls/month (free tier)

```python
def fetch_cryptocompare_news(limit=50):
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        for item in data.get('Data', [])[:limit]:
            print(f"Title: {item['title']}")
            print(f"Source: {item['source']}")
            print(f"URL: {item['url']}")
```

### Google News (RSS)
**Endpoint**: `https://news.google.com/rss/search`
**Auth**: None required
**Format**: RSS Feed

```python
import xml.etree.ElementTree as ET

def fetch_google_news(query, limit=50):
    url = f"https://news.google.com/rss/search?q={query}+cryptocurrency"
    response = requests.get(url)
    
    root = ET.fromstring(response.content)
    for item in root.findall('.//item')[:limit]:
        title = item.find('title').text
        print(f"Title: {title}")
```

### CoinDesk RSS
**Endpoint**: `https://www.coindesk.com/arc/outboundfeeds/rss/`
**Auth**: None required
**Format**: RSS Feed

### Yahoo Finance RSS
**Endpoint**: `https://finance.yahoo.com/rss/topics/crypto`
**Auth**: None required
**Format**: RSS Feed

---

## 5. ENVIRONMENT VARIABLES SETUP

### Create `.env` file in project root:
```bash
# Binance Futures API
BINANCE_API_KEY=Yo5r7wPiEq4fq1pvU2BSynXJes6MPImKWi03S9rzqbySfu4SJP9mbgdPq0T5YAsT
BINANCE_API_SECRET=lgHUKyJ8J1tJk1PhELNlN32KeNdnHJv940IQ5yJg7OoLdH1l7LJLnSQrpJKF9IU0
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret

# Discord Alerts
DISCORD_BOT_TOKEN=2wLqR-vGkcFYILQekphBMRoYlbIvuyqy
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1429173388814844026/16b7qD0GIeCjV0Ul4bwDpSnM4biOz7q56X1mtcF2vQ2QnqOH0wkRv2xAd_HMVZoRNw5O
DISCORD_CHANNEL_ID=1429172043202560163

# Reddit OAuth
REDDIT_CLIENT_ID=oXvmcPz6Sb2ObD5q9FQ0dw
REDDIT_CLIENT_SECRET=RbNtP24dpX_S2t19fbIVTlz-AeZYYA
REDDIT_USER_AGENT=CryptoVolt/3.0 by u/Ill-Database-3830

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cryptovolt

# Redis
REDIS_URL=redis://localhost:6379

# System Config
DEBUG=False
LOG_LEVEL=INFO
PAPER_TRADING_MODE=True  # Set to False for live trading (NOT RECOMMENDED)
```

### Load Environment Variables in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
# ... etc
```

---

## 6. SECURITY BEST PRACTICES

### ✅ DO:
- Store API keys in environment variables
- Use HTTPS for all API calls
- Rotate credentials regularly
- Restrict API key permissions (IP whitelist on Binance)
- Use read-only keys where possible
- Encrypt sensitive data at rest

### ❌ DON'T:
- Commit `.env` to version control
- Hardcode credentials in code
- Share API keys in logs
- Use same keys for test and production
- Give trading permissions to sentiment analysis keys
- Store plaintext secrets in database

### IP Whitelist (Binance)
1. Go to Binance API Management
2. Restrict API key to your server's static IP
3. Disable "Enable Withdrawals"
4. Set trading permissions carefully

---

## 7. ERROR HANDLING & RECOVERY

### Common Errors

#### Binance Errors
```python
# 400 Bad Request - Invalid parameters
# 401 Unauthorized - Wrong API key/secret
# 403 Forbidden - IP not whitelisted
# 429 Too Many Requests - Rate limited
# 500-599 Server errors - Temporary

def handle_binance_error(error_code, error_msg):
    if error_code == 429:
        print("Rate limited. Waiting 60 seconds...")
        time.sleep(60)
    elif error_code in [500, 502, 503]:
        print("Exchange unavailable. Retrying...")
        time.sleep(10)
    else:
        print(f"API Error {error_code}: {error_msg}")
```

#### Network Errors
```python
from requests.exceptions import Timeout, ConnectionError

try:
    response = requests.get(url, timeout=5)
except Timeout:
    print("Request timeout. Retrying...")
except ConnectionError:
    print("Connection error. Check network.")
```

#### Reddit Errors
```python
# Subreddit not found - Check subreddit name
# Authentication failed - Check credentials
# Rate limited - Wait before retrying

def handle_reddit_error(exception):
    error_msg = str(exception)
    if 'private' in error_msg.lower():
        print("Subreddit is private")
    elif 'not found' in error_msg.lower():
        print("Subreddit not found")
    elif 'rate limit' in error_msg.lower():
        print("Rate limited. Waiting...")
        time.sleep(60)
```

---

## 8. TESTING & VALIDATION

### Test API Connectivity
```python
def test_all_apis():
    """Test connectivity to all external APIs"""
    
    print("Testing Binance API...")
    if test_binance_connectivity():
        print("✅ Binance API: OK")
    else:
        print("❌ Binance API: FAILED")
    
    print("Testing Discord Webhook...")
    if test_discord_webhook():
        print("✅ Discord: OK")
    else:
        print("❌ Discord: FAILED")
    
    print("Testing Reddit API...")
    if test_reddit_api():
        print("✅ Reddit: OK")
    else:
        print("❌ Reddit: FAILED")
    
    print("Testing News Sources...")
    if test_news_sources():
        print("✅ News Sources: OK")
    else:
        print("⚠️ Some news sources unavailable")

def test_binance_connectivity():
    try:
        response = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_discord_webhook():
    try:
        payload = {"content": "🧪 CryptoVolt API Test"}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        return response.status_code == 204
    except:
        return False

def test_reddit_api():
    try:
        reddit = praw.Reddit(**REDDIT_CONFIG)
        reddit.redditor('spez').id  # Test with known user
        return True
    except:
        return False

def test_news_sources():
    try:
        response = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
            timeout=10
        )
        return response.status_code == 200
    except:
        return False
```

---

## 9. INTEGRATION CHECKLIST

- [ ] Create `.env` file with all credentials
- [ ] Install required packages: `praw`, `requests`, `python-dotenv`, `vaderSentiment`
- [ ] Test Binance API connectivity
- [ ] Test Discord webhook functionality
- [ ] Test Reddit OAuth authentication
- [ ] Configure IP whitelist on Binance
- [ ] Set environment to PAPER_TRADING_MODE=True
- [ ] Run full API test suite
- [ ] Document any custom configurations
- [ ] Set up logging for API calls
- [ ] Create backup of API credentials (secure location)

---

## 10. MONITORING & LOGGING

### Recommended Logging Setup
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cryptovolt_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info(f"Placing order: {symbol} {side} {quantity}")
logger.warning(f"Rate limit approaching: {remaining} requests left")
logger.error(f"API error: {error_code} {error_msg}")
```

### Metrics to Monitor
- API response times
- Rate limit usage
- Error rates by API
- Message delivery success (Discord)
- Data freshness (latest market/sentiment data)

---

## QUICK REFERENCE

| API | Auth Type | Rate Limit | Purpose |
|-----|-----------|-----------|---------|
| Binance | HMAC SHA256 | 1200 req/min | Trading, Market Data |
| Discord | Webhook | 10 msg/10s | Alerts, Notifications |
| Reddit | OAuth2 | 60 req/min | Sentiment Data |
| News APIs | Various | 100K/month | News Sentiment |

---

**Last Updated**: February 27, 2026  
**Version**: 1.0.0
