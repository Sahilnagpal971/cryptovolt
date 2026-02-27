"""
Enhanced sentiment analysis for CryptoVolt.
Provides Reddit and news aggregation with hybrid sentiment scoring.
"""
import hashlib
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import praw
except ImportError:  # pragma: no cover - optional dependency
    praw = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    FINBERT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FINBERT_AVAILABLE = False


logger = logging.getLogger(__name__)


class EnhancedCryptoSentimentAnalyzer:
    """
    Hybrid sentiment analyzer with:
    - VADER base sentiment
    - Optional FinBERT financial sentiment
    - Crypto lexicon adjustments
    - Sarcasm detection
    - Multi-source aggregation (Reddit + news)
    - In-memory caching
    """

    def __init__(self, reddit_config: Optional[Dict[str, Any]] = None, use_finbert: bool = False, cache_ttl: int = 600):
        self.vader = SentimentIntensityAnalyzer()
        self._enhance_vader_lexicon()

        self.session = self._setup_session()
        self.reddit = self._setup_reddit(reddit_config) if praw else None

        self.use_finbert = bool(use_finbert and FINBERT_AVAILABLE)
        if self.use_finbert:
            self._setup_finbert()
        elif use_finbert and not FINBERT_AVAILABLE:
            logger.warning("FinBERT requested but transformers/torch not installed.")

        self.cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.cache_ttl = cache_ttl

        self.crypto_lexicon = {
            "moon": 3.5,
            "moonshot": 3.0,
            "to the moon": 3.5,
            "parabolic": 3.5,
            "explosive": 3.0,
            "massive pump": 3.5,
            "ath": 3.0,
            "all-time high": 3.0,
            "bullish": 2.5,
            "bull run": 2.8,
            "green candle": 2.0,
            "pump": 2.5,
            "lambo": 2.0,
            "diamond hands": 2.5,
            "hodl": 2.0,
            "accumulate": 2.0,
            "buy the dip": 2.0,
            "breakout": 2.5,
            "rally": 2.5,
            "surge": 2.0,
            "rocket": 2.5,
            "wagmi": 2.0,
            "bullish momentum": 1.5,
            "long": 1.5,
            "uptrend": 1.5,
            "support holds": 1.5,
            "adoption": 1.5,
            "mainstream": 1.5,
            "institutional": 1.5,
            "fomo": 1.0,
            "wen moon": 1.5,
            "gm": 1.0,
            "bullish af": 1.8,
            "rug pull": -4.0,
            "scam": -3.5,
            "ponzi": -3.5,
            "crash": -3.5,
            "collapse": -3.5,
            "liquidated": -3.0,
            "rekt": -3.0,
            "destroyed": -3.0,
            "bearish": -2.5,
            "bear market": -2.8,
            "dump": -2.8,
            "red candle": -2.0,
            "sell-off": -2.5,
            "panic sell": -2.5,
            "correction": -2.0,
            "drop": -2.0,
            "plunge": -2.5,
            "tank": -2.5,
            "paper hands": -2.0,
            "fud": -1.5,
            "short": -1.5,
            "downtrend": -1.5,
            "resistance broken": -1.5,
            "support broken": -2.0,
            "fear": -1.5,
            "capitulation": -2.0,
            "bagholder": -1.5,
            "ngmi": -1.5,
            "whale": 0.5,
            "whales": 0.5,
            "consolidation": 0.0,
            "sideways": 0.0,
            "volatility": -0.3,
            "stable": 0.5,
            "regulation": -0.3,
            "stablecoin": 0.5,
            "dex": 0.3,
            "defi": 0.5,
            "nft": 0.3,
            "web3": 0.5,
            "gwei": 0.0,
            "gas fees": -0.5,
            "degen": 0.3,
            "ser": 0.0,
            "fren": 0.3,
            "anon": 0.0,
        }

        self.sarcasm_patterns = [
            r"(great|awesome|perfect|excellent|wonderful).*?(crash|dump|loss|rekt|liquidated)",
            r"(love|loving|enjoy|enjoying).*?(rekt|liquidated|loss|dump)",
            r"(totally|definitely|absolutely|certainly).*?(not|never|no way)",
            r"sure.*?(crash|dump|bear)",
            r"right.*?(down|crash|loss)",
            r"of course.*?(crash|dump|bear)",
            r"just what we needed.*?(crash|dump|down)",
            r"(bullish|moon).*?(/s|sarcasm)",
            r"keep buying.*?(crash|dump|down)",
        ]

        self.negation_words = {
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nowhere",
            "hardly",
            "scarcely",
            "barely",
        }

    def _setup_reddit(self, config: Optional[Dict[str, Any]]) -> Optional[Any]:
        if not config or not config.get("client_id") or not config.get("client_secret"):
            logger.warning("Reddit OAuth not configured. Using fallback method.")
            return None

        if not praw:
            logger.warning("praw not installed. Using fallback method.")
            return None

        try:
            reddit = praw.Reddit(
                client_id=config.get("client_id"),
                client_secret=config.get("client_secret"),
                user_agent=config.get("user_agent", "CryptoVolt/3.0"),
            )
            reddit.redditor("spez").id
            logger.info("Reddit OAuth authenticated successfully (read-only mode).")
            return reddit
        except Exception as exc:  # pragma: no cover - network dependency
            logger.error("Reddit OAuth failed: %s. Falling back to unauthenticated.", exc)
            return None

    def _setup_finbert(self) -> None:
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_model.eval()
            logger.info("FinBERT model loaded successfully.")
        except Exception as exc:  # pragma: no cover - model download dependency
            logger.error("FinBERT initialization failed: %s", exc)
            self.use_finbert = False

    def _enhance_vader_lexicon(self) -> None:
        crypto_terms = {
            "moon": 3.5,
            "moonshot": 3.0,
            "bullish": 2.5,
            "hodl": 2.0,
            "dump": -2.8,
            "bearish": -2.5,
            "rug pull": -4.0,
            "rekt": -3.0,
            "pump": 2.5,
            "ath": 3.0,
            "fud": -1.5,
            "lambo": 2.0,
            "wagmi": 2.0,
            "ngmi": -1.5,
            "gm": 1.0,
            "diamond hands": 2.5,
            "paper hands": -2.0,
        }
        self.vader.lexicon.update(crypto_terms)

    def _setup_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        param_str = json.dumps(params, sort_keys=True)
        return f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[key]
        return None

    def _set_cache(self, key: str, data: Dict[str, Any]) -> None:
        self.cache[key] = (data, time.time())

    def detect_advanced_sarcasm(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        confidence = 0.0

        for pattern in self.sarcasm_patterns:
            if re.search(pattern, text_lower):
                confidence += 0.3

        sarcasm_markers = ["/s", "sarcasm", "yeah right", "sure thing"]
        if any(marker in text_lower for marker in sarcasm_markers):
            confidence += 0.5

        words = text_lower.split()
        has_negation = any(word in words for word in self.negation_words)

        positive_words = ["good", "great", "awesome", "excellent", "bullish", "moon"]
        has_positive = any(word in text_lower for word in positive_words)

        if has_negation and has_positive:
            confidence += 0.2

        if text.count("!") > 2 or text.count("?") > 2:
            confidence += 0.1

        is_sarcastic = confidence > 0.4
        return is_sarcastic, min(confidence, 1.0)

    def analyze_with_finbert(self, text: str) -> Dict[str, float]:
        if not self.use_finbert:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        try:
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            pos_score = predictions[0][0].item()
            neg_score = predictions[0][1].item()
            neu_score = predictions[0][2].item()
            compound = pos_score - neg_score

            return {
                "compound": compound,
                "pos": pos_score,
                "neg": neg_score,
                "neu": neu_score,
            }
        except Exception as exc:  # pragma: no cover - model runtime dependency
            logger.error("FinBERT analysis error: %s", exc)
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    def hybrid_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 10:
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "confidence": 0.0,
                "is_sarcastic": False,
                "finbert_used": False,
            }

        is_sarcastic, sarcasm_conf = self.detect_advanced_sarcasm(text)
        vader_scores = self.vader.polarity_scores(text)

        if self.use_finbert and len(text) > 30:
            finbert_scores = self.analyze_with_finbert(text)
            base_compound = 0.4 * vader_scores["compound"] + 0.6 * finbert_scores["compound"]
            finbert_used = True
        else:
            finbert_scores = None
            base_compound = vader_scores["compound"]
            finbert_used = False

        crypto_adjustment = self._apply_crypto_lexicon_advanced(text)
        adjusted_compound = base_compound + crypto_adjustment

        if is_sarcastic:
            adjusted_compound = -adjusted_compound * 0.8

        adjusted_compound = max(-1.0, min(1.0, adjusted_compound))

        confidence = self._calculate_sentiment_confidence(
            vader_scores, finbert_scores, len(text), is_sarcastic, sarcasm_conf
        )

        return {
            "compound": adjusted_compound,
            "pos": vader_scores["pos"],
            "neg": vader_scores["neg"],
            "neu": vader_scores["neu"],
            "base_compound": base_compound,
            "crypto_adjustment": crypto_adjustment,
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence": sarcasm_conf,
            "confidence": confidence,
            "finbert_used": finbert_used,
        }

    def _apply_crypto_lexicon_advanced(self, text: str) -> float:
        text_lower = text.lower()
        adjustment = 0.0

        words = text_lower.split()
        negation_indices = [i for i, word in enumerate(words) if word in self.negation_words]

        for term, score in self.crypto_lexicon.items():
            term_lower = term.lower()
            if term_lower in text_lower:
                term_index = text_lower.find(term_lower)
                is_negated = any(abs(term_index - idx * 6) < 30 for idx in negation_indices)

                if is_negated:
                    adjustment -= score * 0.08
                else:
                    adjustment += score * 0.08

        return adjustment

    def _calculate_sentiment_confidence(
        self,
        vader_scores: Dict[str, float],
        finbert_scores: Optional[Dict[str, float]],
        text_length: int,
        is_sarcastic: bool,
        sarcasm_conf: float,
    ) -> float:
        confidence = 0.5

        if text_length > 100:
            confidence += 0.15
        elif text_length > 50:
            confidence += 0.10
        elif text_length > 20:
            confidence += 0.05

        vader_polarity = max(vader_scores["pos"], vader_scores["neg"], vader_scores["neu"])
        confidence += (vader_polarity - vader_scores["neu"]) * 0.1

        if finbert_scores:
            vader_direction = 1 if vader_scores["compound"] > 0 else -1
            finbert_direction = 1 if finbert_scores["compound"] > 0 else -1
            if vader_direction == finbert_direction:
                confidence += 0.15
            else:
                confidence -= 0.10

        if is_sarcastic:
            confidence -= sarcasm_conf * 0.2

        return max(0.0, min(1.0, confidence))

    def calculate_adaptive_weight(self, post: Dict[str, Any], max_score: int = 500, max_comments: int = 200) -> float:
        score = post.get("score", 0)
        comments = post.get("comments", 0)

        score_norm = min(np.log1p(score) / np.log1p(max_score), 1.0)
        comments_norm = min(np.log1p(comments) / np.log1p(max_comments), 1.0)

        post_time = post.get("time", datetime.now())
        hours_ago = (datetime.now() - post_time).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_ago / (14 * 24)))

        author_karma = post.get("author_karma", 0)
        author_cred = min(author_karma / 5000, 1.0) if author_karma else 0.3

        weight = (0.35 * score_norm + 0.35 * comments_norm + 0.20 * recency_score + 0.10 * author_cred)
        return max(0.15, weight)

    def get_reddit_sentiment_praw(self, coin: str = "bitcoin", limit: int = 150) -> Dict[str, Any]:
        if not self.reddit:
            logger.warning("PRAW not available, using fallback.")
            return self.get_reddit_sentiment_fallback(coin, limit)

        cache_key = self._get_cache_key("reddit_praw", {"coin": coin, "limit": limit})
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Using cached Reddit data.")
            return cached

        all_posts: List[Dict[str, Any]] = []
        subreddits_to_search = [coin, "CryptoCurrency", "CryptoMarkets", "Bitcoin", "SatoshiStreetBets"]

        try:
            for subreddit_name in subreddits_to_search:
                if len(all_posts) >= limit:
                    break

                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    search_query = f'"{coin}"'
                    post_stream = list(subreddit.hot(limit=30)) + list(subreddit.new(limit=30)) + list(
                        subreddit.search(search_query, sort="hot", limit=30)
                    )

                    for submission in post_stream:
                        if len(all_posts) >= limit:
                            break

                        if submission.id not in [p.get("id") for p in all_posts]:
                            all_posts.append(
                                {
                                    "id": submission.id,
                                    "title": submission.title,
                                    "selftext": submission.selftext,
                                    "time": datetime.fromtimestamp(submission.created_utc),
                                    "score": submission.score,
                                    "comments": submission.num_comments,
                                    "subreddit": subreddit_name,
                                    "author_karma": submission.author.comment_karma
                                    if (submission.author and hasattr(submission.author, "comment_karma"))
                                    else 0,
                                    "engagement": submission.score + submission.num_comments,
                                }
                            )

                    time.sleep(1)

                except Exception as exc:  # pragma: no cover - external API dependency
                    logger.warning("Subreddit %s error: %s", subreddit_name, exc)
                    continue

            result = self._analyze_posts(all_posts, limit, "Reddit (PRAW)")
            self._set_cache(cache_key, result)
            return result

        except Exception as exc:  # pragma: no cover - external API dependency
            logger.error("PRAW error: %s", exc)
            return self.default_sentiment("Reddit (PRAW)")

    def get_reddit_sentiment_fallback(self, coin: str = "bitcoin", limit: int = 150) -> Dict[str, Any]:
        cache_key = self._get_cache_key("reddit_fallback", {"coin": coin, "limit": limit})
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Using cached Reddit data.")
            return cached

        all_posts: List[Dict[str, Any]] = []
        subreddits = [coin, "CryptoCurrency", "CryptoMarkets", "ethereum"]
        endpoints = ["hot", "new"]
        headers = {"User-Agent": "CryptoVolt/3.0"}

        try:
            for subreddit in subreddits:
                if len(all_posts) >= limit:
                    break

                for endpoint in endpoints:
                    if len(all_posts) >= limit:
                        break

                    url = f"https://www.reddit.com/r/{subreddit}/{endpoint}.json?limit=100"

                    try:
                        response = self.session.get(url, headers=headers, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            posts = data.get("data", {}).get("children", [])

                            for post in posts:
                                if len(all_posts) >= limit:
                                    break

                                post_data = post["data"]
                                title = post_data.get("title", "")

                                if title and len(title) > 10:
                                    if title not in [p.get("title") for p in all_posts]:
                                        all_posts.append(
                                            {
                                                "title": title,
                                                "selftext": post_data.get("selftext", ""),
                                                "time": datetime.fromtimestamp(post_data.get("created_utc", 0)),
                                                "score": post_data.get("score", 0),
                                                "comments": post_data.get("num_comments", 0),
                                                "subreddit": subreddit,
                                                "author_karma": 0,
                                                "engagement": post_data.get("score", 0) + post_data.get("num_comments", 0),
                                            }
                                        )

                            time.sleep(1.5)

                    except Exception as exc:  # pragma: no cover - external API dependency
                        logger.warning("Fetch error for r/%s: %s", subreddit, exc)
                        continue

            result = self._analyze_posts(all_posts, limit, "Reddit (Fallback)")
            self._set_cache(cache_key, result)
            return result

        except Exception as exc:  # pragma: no cover - external API dependency
            logger.error("Reddit fallback error: %s", exc)
            return self.default_sentiment("Reddit (Fallback)")

    def get_news_sentiment_enhanced(self, coin_id: str = "BTC", coin_name: str = "bitcoin", limit: int = 150) -> Dict[str, Any]:
        cache_key = self._get_cache_key("news", {"coin_id": coin_id, "coin_name": coin_name, "limit": limit})
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Using cached news data.")
            return cached

        sources = [
            ("CryptoCompare", self._fetch_cryptocompare_news(limit // 3)),
            ("Google News", self._fetch_google_news(coin_name, limit // 3)),
            ("CoinDesk", self._fetch_coindesk_news(limit // 4)),
            ("Yahoo Finance", self._fetch_yahoo_news(limit // 4)),
        ]

        all_news_tuples: List[Tuple[str, str]] = []
        for source_name, articles in sources:
            for article_title in articles:
                all_news_tuples.append((article_title, source_name))

        unique_news = list(dict.fromkeys(all_news_tuples))[:limit]

        news_posts = [
            {
                "title": headline,
                "selftext": "",
                "time": datetime.now(),
                "score": 1,
                "comments": 0,
                "subreddit": source,
                "author_karma": 0,
                "engagement": 1,
            }
            for (headline, source) in unique_news
        ]

        result = self._analyze_posts(news_posts, limit, "News (Multi-Source)")
        self._set_cache(cache_key, result)
        return result

    def _fetch_cryptocompare_news(self, limit: int) -> List[str]:
        news_items: List[str] = []
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("Data", [])[:limit]:
                    title = item.get("title", "")
                    if title:
                        news_items.append(title)
        except Exception as exc:  # pragma: no cover - external API dependency
            logger.warning("CryptoCompare error: %s", exc)
        return news_items

    def _fetch_google_news(self, query: str, limit: int) -> List[str]:
        news_items: List[str] = []
        try:
            url = f"https://news.google.com/rss/search?q={query}+cryptocurrency&hl=en-US&gl=US&ceid=US:en"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title")
                    if title is not None and title.text:
                        news_items.append(title.text)
        except Exception as exc:  # pragma: no cover - external API dependency
            logger.warning("Google News error: %s", exc)
        return news_items

    def _fetch_coindesk_news(self, limit: int) -> List[str]:
        news_items: List[str] = []
        try:
            url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title")
                    if title is not None and title.text:
                        news_items.append(title.text)
        except Exception as exc:  # pragma: no cover - external API dependency
            logger.warning("CoinDesk error: %s", exc)
        return news_items

    def _fetch_yahoo_news(self, limit: int) -> List[str]:
        news_items: List[str] = []
        try:
            url = "https://finance.yahoo.com/rss/topics/crypto"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title")
                    if title is not None and title.text:
                        news_items.append(title.text)
        except Exception as exc:  # pragma: no cover - external API dependency
            logger.warning("Yahoo Finance error: %s", exc)
        return news_items

    def _analyze_posts(self, posts: List[Dict[str, Any]], target_limit: int, source_name: str) -> Dict[str, Any]:
        if not posts:
            logger.warning("No posts to analyze for %s", source_name)
            return self.default_sentiment(source_name)

        sentiments: List[Dict[str, Any]] = []
        weighted_sentiments: List[Tuple[float, float]] = []
        sentiment_by_sub: Dict[str, List[float]] = defaultdict(list)

        for post in posts:
            text = f"{post['title']} {post.get('selftext', '')[:200]}"
            sentiment = self.hybrid_sentiment_analysis(text)

            if "News" in source_name:
                weight = 1.0
            else:
                weight = self.calculate_adaptive_weight(post)

            sentiments.append(sentiment)
            weighted_sentiments.append((sentiment["compound"], weight))
            sentiment_by_sub[post.get("subreddit", "news")].append(sentiment["compound"])

        all_compounds = [s["compound"] for s in sentiments]
        avg_unweighted_score = statistics.mean(all_compounds)
        std_dev = statistics.stdev(all_compounds) if len(all_compounds) > 1 else 0.0

        total_weight = sum(w for _, w in weighted_sentiments)
        if total_weight == 0:
            avg_weighted_score = avg_unweighted_score
        else:
            avg_weighted_score = sum(s * w for s, w in weighted_sentiments) / total_weight

        if avg_weighted_score > 0.05:
            sentiment_label = "BULLISH"
        elif avg_weighted_score < -0.05:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        reliability = min(len(posts) / target_limit, 1.0)
        avg_sentiment_confidence = statistics.mean(s["confidence"] for s in sentiments)

        pos_count = sum(1 for s in all_compounds if s > 0.05)
        neg_count = sum(1 for s in all_compounds if s < -0.05)
        neu_count = len(all_compounds) - pos_count - neg_count
        sarcasm_count = sum(1 for s in sentiments if s["is_sarcastic"])
        finbert_count = sum(1 for s in sentiments if s["finbert_used"])

        disaggregation = {sub: round(statistics.mean(scores), 4) for sub, scores in sentiment_by_sub.items()}

        return {
            "sentiment": sentiment_label,
            "weighted_score": round(avg_weighted_score, 4),
            "unweighted_score": round(avg_unweighted_score, 4),
            "reliability": round(reliability, 2),
            "std_dev": round(std_dev, 4),
            "confidence": round(avg_sentiment_confidence, 2),
            "sample_size": len(posts),
            "target_sample": target_limit,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "neu_count": neu_count,
            "sarcasm_count": sarcasm_count,
            "finbert_count": finbert_count,
            "source_name": source_name,
            "disaggregation": disaggregation,
        }

    def default_sentiment(self, source_name: str = "N/A") -> Dict[str, Any]:
        return {
            "sentiment": "NEUTRAL",
            "weighted_score": 0.0,
            "unweighted_score": 0.0,
            "reliability": 0.0,
            "std_dev": 0.0,
            "confidence": 0.0,
            "sample_size": 0,
            "target_sample": 0,
            "pos_count": 0,
            "neg_count": 0,
            "neu_count": 0,
            "sarcasm_count": 0,
            "finbert_count": 0,
            "source_name": source_name,
            "disaggregation": {},
        }

    def get_combined_market_sentiment(
        self,
        coin: str = "bitcoin",
        coin_id: str = "BTC",
        reddit_limit: int = 150,
        news_limit: int = 150,
    ) -> Dict[str, Any]:
        reddit_data = self.get_reddit_sentiment_praw(coin, reddit_limit)
        news_data = self.get_news_sentiment_enhanced(coin_id, coin, news_limit)

        total_samples = reddit_data["sample_size"] + news_data["sample_size"]
        if total_samples == 0:
            return {
                "final_sentiment": "NEUTRAL",
                "final_score": 0.0,
                "final_confidence": 0.0,
                "total_samples": 0,
                "source_divergence": {"score": 0.0, "level": "N/A"},
                "reddit_analysis": reddit_data,
                "news_analysis": news_data,
            }

        r_weight = reddit_data["sample_size"] * reddit_data["reliability"]
        n_weight = news_data["sample_size"] * news_data["reliability"]
        total_weight = r_weight + n_weight

        if total_weight == 0:
            final_score = 0.0
            final_confidence = 0.0
        else:
            final_score = (
                (reddit_data["weighted_score"] * r_weight) + (news_data["weighted_score"] * n_weight)
            ) / total_weight

            final_confidence = (
                (reddit_data["confidence"] * r_weight) + (news_data["confidence"] * n_weight)
            ) / total_weight

        if final_score > 0.05:
            final_sentiment = "BULLISH"
        elif final_score < -0.05:
            final_sentiment = "BEARISH"
        else:
            final_sentiment = "NEUTRAL"

        divergence_score = abs(reddit_data["weighted_score"] - news_data["weighted_score"])
        if divergence_score > 0.5:
            divergence_level = "HIGH"
        elif divergence_score > 0.2:
            divergence_level = "MEDIUM"
        else:
            divergence_level = "LOW"

        return {
            "final_sentiment": final_sentiment,
            "final_score": round(final_score, 4),
            "final_confidence": round(final_confidence, 2),
            "total_samples": total_samples,
            "source_divergence": {"score": round(divergence_score, 4), "level": divergence_level},
            "reddit_analysis": reddit_data,
            "news_analysis": news_data,
        }
