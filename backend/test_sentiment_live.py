"""
Test sentiment analysis system with real data from Reddit and News
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from app.sentiment.analyzer import EnhancedCryptoSentimentAnalyzer
from app.core.config import settings
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_sentiment():
    """Test sentiment analysis on sample texts"""
    print("\n" + "=" * 60)
    print("Testing Text Sentiment Analysis")
    print("=" * 60 + "\n")
    
    analyzer = EnhancedCryptoSentimentAnalyzer()
    
    test_texts = [
        ("Bitcoin to the moon! 🚀 Great buy opportunity!", "BULLISH"),
        ("Market crash imminent. Sell everything now!", "BEARISH"),
        ("BTC holding steady at support levels", "NEUTRAL"),
        ("Ethereum upgrade looking promising. Bullish long term", "BULLISH"),
        ("Dump incoming! Whales are selling", "BEARISH"),
    ]
    
    for text, expected in test_texts:
        result = analyzer.hybrid_sentiment_analysis(text)
        
        # Convert compound score to sentiment label
        compound = result['compound']
        if compound > 0.05:
            sentiment = "BULLISH"
        elif compound < -0.05:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        print(f"Text: {text[:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Result: {sentiment} (compound: {compound:.3f})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Crypto adjustment: {result.get('crypto_adjustment', 0):.3f}")
        print()


def test_reddit_sentiment():
    """Test Reddit sentiment fetching"""
    print("\n" + "=" * 60)
    print("Testing Reddit Sentiment Analysis")
    print("=" * 60 + "\n")
    
    analyzer = EnhancedCryptoSentimentAnalyzer()
    
    coins = ["bitcoin", "ethereum"]
    
    for coin in coins:
        print(f"\nAnalyzing Reddit sentiment for {coin.upper()}...")
        try:
            result = analyzer.get_reddit_sentiment_praw(coin, limit=50)
            
            print(f"  Posts analyzed: {result.get('post_count', 0)}")
            print(f"  Overall sentiment: {result.get('overall_sentiment', 'N/A')}")
            print(f"  Sentiment score: {result.get('sentiment_score', 0):.3f}")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")
            print(f"  Avg engagement: {result.get('avg_engagement', 0):.1f}")
            
            if result.get('top_posts'):
                print(f"  Top posts:")
                for post in result['top_posts'][:3]:
                    print(f"    - {post.get('title', '')[:60]}... (score: {post.get('sentiment_score', 0):.2f})")
        
        except Exception as e:
            logger.error(f"Error fetching Reddit data for {coin}: {e}")
            print(f"  ❌ Error: {str(e)}")


def test_news_sentiment():
    """Test news sentiment fetching"""
    print("\n" + "=" * 60)
    print("Testing News Sentiment Analysis")
    print("=" * 60 + "\n")
    
    analyzer = EnhancedCryptoSentimentAnalyzer()
    
    coins = [
        ("bitcoin", "BTC"),
        ("ethereum", "ETH"),
    ]
    
    for coin_name, coin_id in coins:
        print(f"\nAnalyzing news sentiment for {coin_name.upper()}...")
        try:
            result = analyzer.get_news_sentiment_enhanced(coin_id, coin_name, limit=50)
            
            print(f"  Articles analyzed: {result.get('article_count', 0)}")
            print(f"  Overall sentiment: {result.get('overall_sentiment', 'N/A')}")
            print(f"  Sentiment score: {result.get('sentiment_score', 0):.3f}")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")
            
            if result.get('top_articles'):
                print(f"  Top articles:")
                for article in result['top_articles'][:3]:
                    print(f"    - {article.get('title', '')[:60]}... (score: {article.get('sentiment_score', 0):.2f})")
        
        except Exception as e:
            logger.error(f"Error fetching news data for {coin_name}: {e}")
            print(f"  ❌ Error: {str(e)}")


def test_combined_sentiment():
    """Test combined sentiment analysis (Reddit + News)"""
    print("\n" + "=" * 60)
    print("Testing Combined Sentiment Analysis")
    print("=" * 60 + "\n")
    
    analyzer = EnhancedCryptoSentimentAnalyzer()
    
    print("Analyzing combined sentiment for Bitcoin...")
    try:
        result = analyzer.get_combined_market_sentiment(
            coin="bitcoin",
            coin_id="BTC",
            reddit_limit=50,
            news_limit=50
        )
        
        print(f"\n📊 Combined Analysis Results:")
        print(f"  Final Sentiment: {result.get('final_sentiment', 'N/A')}")
        print(f"  Final Score: {result.get('final_score', 0):.3f}")
        print(f"  Confidence: {result.get('final_confidence', 0):.2%}")
        
        print(f"\n  Reddit Component:")
        reddit = result.get('reddit_data', {})
        print(f"    Sentiment: {reddit.get('overall_sentiment', 'N/A')}")
        print(f"    Score: {reddit.get('sentiment_score', 0):.3f}")
        print(f"    Posts: {reddit.get('post_count', 0)}")
        
        print(f"\n  News Component:")
        news = result.get('news_data', {})
        print(f"    Sentiment: {news.get('overall_sentiment', 'N/A')}")
        print(f"    Score: {news.get('sentiment_score', 0):.3f}")
        print(f"    Articles: {news.get('article_count', 0)}")
        
        print(f"\n  Source Divergence: {result.get('source_divergence', 0):.3f}")
        if result.get('source_divergence', 0) > 0.3:
            print(f"    ⚠️  High divergence between sources - use caution!")
        
        # Save results for reference
        with open("sentiment_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Results saved to sentiment_test_results.json")
    
    except Exception as e:
        logger.error(f"Error in combined sentiment analysis: {e}")
        print(f"  ❌ Error: {str(e)}")


def main():
    """Run all sentiment tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "CryptoVolt Sentiment Analysis Testing")
    print("=" * 70)
    
    # Check if credentials are configured
    if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
        print("\n⚠️  Reddit credentials not configured in .env file")
        print("   Some tests may fail. Configure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
    
    # Run tests
    test_text_sentiment()
    test_reddit_sentiment()
    test_news_sentiment()
    test_combined_sentiment()
    
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
