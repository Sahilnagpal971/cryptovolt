"""
Sentiment Analysis Service
Processes news and social media data for sentiment scores
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentSource(str, Enum):
    """Sentiment data sources"""
    REDDIT = "REDDIT"
    TWITTER = "TWITTER"
    NEWS = "NEWS"
    COMBINED = "COMBINED"


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer
    Aggregates sentiment from news, Reddit, Twitter
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sentiment analyzer"""
        self.config = config or {}
        self.sources_enabled = self.config.get("sources", ["NEWS", "REDDIT", "TWITTER"])
        self.aggregation_method = self.config.get("aggregation_method", "weighted_average")
        
    def analyze_symbol_sentiment(
        self,
        symbol: str,
        sentiment_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze overall sentiment for a symbol
        
        Args:
            symbol: Trading pair symbol
            sentiment_data: List of sentiment records with scores
            
        Returns:
            Aggregated sentiment analysis
        """
        
        if not sentiment_data:
            return {
                "symbol": symbol,
                "overall_score": 0.0,
                "strength": 0.0,
                "trend": "NEUTRAL",
                "sources_analyzed": 0,
                "timestamp": datetime.utcnow(),
            }
        
        # Aggregate scores
        scores_by_source = self._group_by_source(sentiment_data)
        aggregated_scores = {}
        
        for source, records in scores_by_source.items():
            scores = [r.get("sentiment_score", 0) for r in records]
            aggregated_scores[source] = {
                "mean": sum(scores) / len(scores) if scores else 0,
                "count": len(records),
                "recent": sorted(records, key=lambda x: x.get("timestamp", ""))[-1:],
            }
        
        # Calculate overall sentiment
        overall_score = self._aggregate_scores(aggregated_scores)
        
        # Determine trend
        trend = self._determine_trend(overall_score)
        
        # Calculate strength (confidence in sentiment signal)
        strength = self._calculate_sentiment_strength(aggregated_scores)
        
        return {
            "symbol": symbol,
            "overall_score": overall_score,  # -1 to 1
            "strength": strength,  # 0 to 1
            "trend": trend,
            "sources_analyzed": sum(score["count"] for score in aggregated_scores.values()),
            "by_source": aggregated_scores,
            "timestamp": datetime.utcnow(),
        }
    
    def _group_by_source(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group sentiment data by source"""
        grouped = {}
        for record in sentiment_data:
            source = record.get("source", "UNKNOWN")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(record)
        return grouped
    
    def _aggregate_scores(self, scores_by_source: Dict[str, Any]) -> float:
        """Aggregate scores across sources using configured method"""
        if not scores_by_source:
            return 0.0
        
        if self.aggregation_method == "weighted_average":
            # Weight by source importance
            weights = {
                "NEWS": 0.4,
                "REDDIT": 0.35,
                "TWITTER": 0.25,
            }
            
            total_weight = 0
            weighted_sum = 0
            
            for source, data in scores_by_source.items():
                weight = weights.get(source, 0.2)
                weighted_sum += data["mean"] * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.aggregation_method == "simple_average":
            scores = [data["mean"] for data in scores_by_source.values()]
            return sum(scores) / len(scores) if scores else 0.0
        
        elif self.aggregation_method == "median":
            scores = [data["mean"] for data in scores_by_source.values()]
            scores.sort()
            mid = len(scores) // 2
            return scores[mid] if scores else 0.0
        
        return 0.0
    
    def _determine_trend(self, score: float) -> str:
        """Determine sentiment trend based on score"""
        if score > 0.3:
            return "VERY_POSITIVE"
        elif score > 0.1:
            return "POSITIVE"
        elif score > -0.1:
            return "NEUTRAL"
        elif score > -0.3:
            return "NEGATIVE"
        else:
            return "VERY_NEGATIVE"
    
    def _calculate_sentiment_strength(self, scores_by_source: Dict[str, Any]) -> float:
        """
        Calculate strength (confidence) of sentiment signal
        Based on agreement across sources
        """
        if not scores_by_source:
            return 0.0
        
        scores = [abs(data["mean"]) for data in scores_by_source.values()]
        average_confidence = sum(scores) / len(scores) if scores else 0
        
        # Bonus for multiple sources agreeing
        sources_count = len(scores_by_source)
        agreement_bonus = min(sources_count / 3, 0.2)  # Max +0.2 for agreement
        
        return min(average_confidence + agreement_bonus, 1.0)


class SentimentFetcher:
    """
    Fetches sentiment data from various sources
    Implements sampling plan from SRS (Appendix C)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sentiment fetcher"""
        self.config = config or {}
        # SRS Sampling Plan: ~300 items per window (150 Reddit + 150 News)
        self.reddit_sample_size = self.config.get("reddit_sample_size", 150)
        self.news_sample_size = self.config.get("news_sample_size", 150)
        
    async def fetch_sentiment_batch(
        self,
        symbol: str,
        sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch sentiment data in batches
        Implements SRS documented sampling plan
        """
        if sources is None:
            sources = ["REDDIT", "NEWS"]
        
        batch = []
        
        # Fetch from each source
        for source in sources:
            if source == "REDDIT":
                data = await self._fetch_reddit_sentiment(symbol)
                batch.extend(data)
            elif source == "NEWS":
                data = await self._fetch_news_sentiment(symbol)
                batch.extend(data)
            elif source == "TWITTER":
                data = await self._fetch_twitter_sentiment(symbol)
                batch.extend(data)
        
        return batch
    
    async def _fetch_reddit_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch sentiment from Reddit (limit to sample size)"""
        # TODO: Implement Reddit API integration
        logger.info(f"Fetching Reddit sentiment for {symbol}")
        return []
    
    async def _fetch_news_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch sentiment from news sources (limit to sample size)"""
        # TODO: Implement news API integration
        logger.info(f"Fetching news sentiment for {symbol}")
        return []
    
    async def _fetch_twitter_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch sentiment from Twitter/X"""
        # TODO: Implement Twitter API integration
        logger.info(f"Fetching Twitter sentiment for {symbol}")
        return []
