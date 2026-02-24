# Processing Commodity News for Sentiment Analysis

## In Brief

Commodity news processing involves acquiring real-time news from multiple sources (RSS feeds, Twitter, Bloomberg terminals), filtering for relevance, extracting key information (commodities mentioned, price movements, supply/demand indicators), and preparing text for sentiment analysis while handling market-specific language and time-critical information flow.

> 💡 **Key Insight:** Commodity news is fundamentally different from general financial news—a single word change ("higher" vs "lower" production forecast) can move markets billions of dollars. Processing must be low-latency (sub-minute for algorithmic trading), handle domain-specific jargon ("contango", "crack spread"), and distinguish between analysis and reporting of facts.

## Formal Definition

A commodity news processing pipeline is a function **N: S → E** where:

**Input:** S = stream of raw news from sources {RSS, API, web scraping}

**Output:** E = enriched news events with structure:
```
Event = {
  event_id: unique identifier,
  timestamp: publication time (UTC),
  source: {publisher, author, credibility_score},
  content: {headline, summary, full_text},
  entities: {
    commodities: [(commodity, mention_count, context)],
    companies: [(name, ticker, role)],
    geographies: [(location, relevance)],
    metrics: [(metric_type, value, unit, change_direction)]
  },
  sentiment: {polarity, confidence, rationale},
  urgency: priority_score,
  market_impact: predicted_relevance
}
```

**Pipeline stages:**
1. **Acquisition**: Pull from sources at frequency (1min - 1hr)
2. **Deduplication**: Detect same story from multiple sources
3. **Relevance filtering**: Keep only commodity-relevant news
4. **Entity extraction**: Identify commodities, companies, locations
5. **Fact extraction**: Parse production numbers, forecasts, events
6. **Sentiment tagging**: Classify market sentiment
7. **Prioritization**: Score by likely market impact

## Intuitive Explanation

Think of processing commodity news like being a trader's research assistant who reads thousands of articles per day:

**Your job:**
- Monitor dozens of news sources constantly
- Immediately flag anything about your commodities (crude oil, corn, etc.)
- Ignore duplicate stories (same Reuters story on 10 different sites)
- Extract the key facts: "Saudi Arabia cuts production by 500k bpd"
- Determine if it's bullish (positive for prices) or bearish (negative)
- Rank by importance: OPEC production cuts > local refinery maintenance

**Challenges:**
- Speed matters: News moves markets in seconds
- Context matters: "Increased production" is bearish for prices but might be bullish for producer stocks
- Jargon: "Contango steepens" has specific meaning in commodities
- Conflicting reports: Different sources give different numbers

## Code Implementation

### Multi-Source News Acquisition

```python
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import tweepy
import hashlib
import time

class NewsSource(Enum):
    RSS_FEED = "rss"
    TWITTER = "twitter"
    NEWS_API = "news_api"
    BLOOMBERG_TERMINAL = "bloomberg"  # Requires subscription
    SPECIALIZED_FEED = "specialized"

@dataclass
class RawNewsItem:
    """Raw news item from source."""
    source: NewsSource
    source_name: str
    headline: str
    summary: str
    full_text: Optional[str]
    url: str
    published_at: datetime
    author: Optional[str]
    tags: List[str]

class CommodityNewsAcquisition:
    """
    Acquire commodity news from multiple sources.
    """

    def __init__(self):
        # Commodity-focused RSS feeds
        self.rss_feeds = {
            "energy": [
                "https://www.rigzone.com/news/rss.aspx",
                "https://www.naturalgasintel.com/feed/",
                "https://www.oilprice.com/rss/main",
                "https://www.reuters.com/rssFeed/energy"
            ],
            "agriculture": [
                "https://www.agriculture.com/rss",
                "https://www.farms.com/ag-news-feed/",
                "https://www.reuters.com/rssFeed/agriculture"
            ],
            "metals": [
                "https://www.mining.com/rss/",
                "https://www.kitco.com/rss/KitcoNews.xml",
                "https://www.reuters.com/rssFeed/metals"
            ]
        }

        # Track seen articles to avoid duplicates
        self.seen_hashes = set()
        self.hash_expiry = timedelta(hours=24)
        self.hash_timestamps = {}

    def fetch_rss_feeds(self, commodity_type: str) -> List[RawNewsItem]:
        """
        Fetch news from RSS feeds for commodity type.
        """
        items = []

        for feed_url in self.rss_feeds.get(commodity_type, []):
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()

                    # Create raw item
                    item = RawNewsItem(
                        source=NewsSource.RSS_FEED,
                        source_name=feed.feed.get('title', 'Unknown'),
                        headline=entry.title,
                        summary=entry.get('summary', ''),
                        full_text=None,  # RSS usually doesn't include full text
                        url=entry.link,
                        published_at=pub_date,
                        author=entry.get('author'),
                        tags=[]
                    )

                    items.append(item)

            except Exception as e:
                print(f"Error fetching {feed_url}: {e}")

        return items

    def fetch_news_api(
        self,
        query: str,
        api_key: str,
        hours_back: int = 24
    ) -> List[RawNewsItem]:
        """
        Fetch news from NewsAPI (newsapi.org).

        Args:
            query: Search query (e.g., "crude oil" OR "petroleum")
            api_key: NewsAPI key
            hours_back: How far back to search
        """
        url = "https://newsapi.org/v2/everything"

        from_date = datetime.now() - timedelta(hours=hours_back)

        params = {
            'q': query,
            'from': from_date.isoformat(),
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'language': 'en'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        items = []
        for article in data.get('articles', []):
            item = RawNewsItem(
                source=NewsSource.NEWS_API,
                source_name=article['source']['name'],
                headline=article['title'],
                summary=article['description'] or '',
                full_text=article.get('content'),
                url=article['url'],
                published_at=datetime.fromisoformat(
                    article['publishedAt'].replace('Z', '+00:00')
                ),
                author=article.get('author'),
                tags=[]
            )
            items.append(item)

        return items

    def fetch_twitter(
        self,
        keywords: List[str],
        twitter_bearer_token: str,
        max_results: int = 100
    ) -> List[RawNewsItem]:
        """
        Fetch commodity-related tweets.

        Focus on:
        - Official accounts (EIA, USDA, companies)
        - Verified analysts
        - Breaking news
        """
        client = tweepy.Client(bearer_token=twitter_bearer_token)

        # Build query
        query = " OR ".join(keywords) + " -is:retweet lang:en"

        # Search recent tweets
        tweets = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['created_at', 'author_id', 'public_metrics']
        )

        items = []
        if tweets.data:
            for tweet in tweets.data:
                item = RawNewsItem(
                    source=NewsSource.TWITTER,
                    source_name="Twitter",
                    headline=tweet.text[:100],  # First 100 chars as headline
                    summary=tweet.text,
                    full_text=tweet.text,
                    url=f"https://twitter.com/user/status/{tweet.id}",
                    published_at=tweet.created_at,
                    author=str(tweet.author_id),
                    tags=keywords
                )
                items.append(item)

        return items

    def deduplicate(self, items: List[RawNewsItem]) -> List[RawNewsItem]:
        """
        Remove duplicate news items.

        Duplicates are detected by:
        1. Identical URLs
        2. Similar headlines (fuzzy matching)
        3. Content hash (for syndicated content)
        """
        # Clean up expired hashes
        self._clean_expired_hashes()

        unique_items = []

        for item in items:
            # Create content hash
            content = f"{item.headline}|{item.summary[:200]}"
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Check if seen before
            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                self.hash_timestamps[content_hash] = datetime.now()
                unique_items.append(item)

        return unique_items

    def _clean_expired_hashes(self):
        """Remove hashes older than expiry time."""
        now = datetime.now()
        expired = [
            h for h, ts in self.hash_timestamps.items()
            if now - ts > self.hash_expiry
        ]

        for h in expired:
            self.seen_hashes.discard(h)
            del self.hash_timestamps[h]
```

### Commodity Relevance Filtering

```python
from anthropic import Anthropic

class RelevanceFilter:
    """
    Filter news for commodity relevance using LLM.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def is_relevant(
        self,
        item: RawNewsItem,
        target_commodities: List[str]
    ) -> tuple[bool, float, str]:
        """
        Determine if news item is relevant to target commodities.

        Returns: (is_relevant, confidence, reasoning)
        """
        prompt = f"""Determine if this news article is relevant to commodity markets: {', '.join(target_commodities)}.

Headline: {item.headline}
Summary: {item.summary}

Return JSON:
{{
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "mentioned_commodities": ["crude_oil", "gasoline", ...],
  "reasoning": "Brief explanation of why relevant/not relevant"
}}

Consider relevant if:
- Directly mentions the commodities
- Discusses supply/demand factors (production, inventory, consumption)
- Covers geopolitical events affecting these commodities
- Involves major producers/consumers
- Discusses related markets (e.g., refining for crude oil)

Consider NOT relevant if:
- Only tangentially related
- About individual company stock movements (unless major producer)
- General economic news without commodity connection"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        result = json.loads(response.content[0].text)

        return (
            result["is_relevant"],
            result["confidence"],
            result["reasoning"]
        )

    def batch_filter(
        self,
        items: List[RawNewsItem],
        target_commodities: List[str],
        min_confidence: float = 0.7
    ) -> List[RawNewsItem]:
        """
        Filter a batch of news items for relevance.
        """
        relevant_items = []

        for item in items:
            is_rel, conf, reasoning = self.is_relevant(item, target_commodities)

            if is_rel and conf >= min_confidence:
                # Add metadata
                item.tags.extend(['relevant', f'confidence:{conf:.2f}'])
                relevant_items.append(item)

        return relevant_items
```

### Entity and Fact Extraction

```python
@dataclass
class ExtractedEntities:
    """Entities extracted from news."""
    commodities: List[Dict[str, any]]  # {commodity, mention_count, context}
    companies: List[Dict[str, str]]  # {name, ticker, role}
    geographies: List[str]
    metrics: List[Dict[str, any]]  # {metric, value, unit, direction}
    events: List[str]  # Specific events mentioned

class EntityExtractor:
    """
    Extract commodity-specific entities from news.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def extract_entities(self, item: RawNewsItem) -> ExtractedEntities:
        """
        Extract structured entities from news item.
        """
        text = f"{item.headline}\n\n{item.summary}"

        prompt = f"""Extract commodity market entities from this news article.

Article:
{text}

Return JSON with this structure:
{{
  "commodities": [
    {{"name": "crude_oil", "mention_count": 3, "context": "production increase"}},
    {{"name": "gasoline", "mention_count": 1, "context": "demand"}}
  ],
  "companies": [
    {{"name": "Saudi Aramco", "ticker": null, "role": "producer"}},
    {{"name": "ExxonMobil", "ticker": "XOM", "role": "producer/refiner"}}
  ],
  "geographies": ["Saudi Arabia", "United States", "OPEC"],
  "metrics": [
    {{
      "metric": "production",
      "value": 500000,
      "unit": "barrels_per_day",
      "direction": "increase",
      "commodity": "crude_oil"
    }},
    {{
      "metric": "inventory",
      "value": 5.2,
      "unit": "million_barrels",
      "direction": "decrease",
      "commodity": "crude_oil"
    }}
  ],
  "events": [
    "OPEC production cut announcement",
    "Refinery maintenance scheduled"
  ]
}}

Extract specific numbers where mentioned. Use standard units:
- Production: barrels_per_day (bpd), million_barrels_per_day (mbpd)
- Inventory: million_barrels (mmb), billion_cubic_feet (bcf)
- Prices: dollars_per_barrel, dollars_per_mmbtu"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        extracted = json.loads(response.content[0].text)

        return ExtractedEntities(
            commodities=extracted["commodities"],
            companies=extracted["companies"],
            geographies=extracted["geographies"],
            metrics=extracted["metrics"],
            events=extracted["events"]
        )
```

### Complete News Processing Pipeline

```python
@dataclass
class ProcessedNewsEvent:
    """Fully processed news event ready for sentiment analysis."""
    event_id: str
    timestamp: datetime
    source: str
    headline: str
    summary: str
    url: str
    entities: ExtractedEntities
    priority_score: float  # 0-1, based on likely market impact

class CommodityNewsProcessor:
    """
    End-to-end news processing pipeline.
    """

    def __init__(self, anthropic_api_key: str, news_api_key: Optional[str] = None):
        self.acquisition = CommodityNewsAcquisition()
        self.relevance_filter = RelevanceFilter(anthropic_api_key)
        self.entity_extractor = EntityExtractor(anthropic_api_key)
        self.news_api_key = news_api_key

    def process_news_stream(
        self,
        commodity_type: str,
        target_commodities: List[str],
        hours_back: int = 1
    ) -> List[ProcessedNewsEvent]:
        """
        Process news stream for commodity type.

        Args:
            commodity_type: "energy", "agriculture", "metals"
            target_commodities: Specific commodities to focus on
            hours_back: How far back to fetch news

        Returns:
            List of processed, prioritized news events
        """
        # Step 1: Acquire from multiple sources
        raw_items = []

        # RSS feeds
        rss_items = self.acquisition.fetch_rss_feeds(commodity_type)
        raw_items.extend(rss_items)

        # News API (if available)
        if self.news_api_key:
            query = " OR ".join(target_commodities)
            api_items = self.acquisition.fetch_news_api(
                query,
                self.news_api_key,
                hours_back=hours_back
            )
            raw_items.extend(api_items)

        print(f"Acquired {len(raw_items)} raw news items")

        # Step 2: Deduplicate
        unique_items = self.acquisition.deduplicate(raw_items)
        print(f"After deduplication: {len(unique_items)} unique items")

        # Step 3: Filter for relevance
        relevant_items = self.relevance_filter.batch_filter(
            unique_items,
            target_commodities,
            min_confidence=0.7
        )
        print(f"After relevance filtering: {len(relevant_items)} relevant items")

        # Step 4: Extract entities and create processed events
        processed_events = []

        for item in relevant_items:
            # Extract entities
            entities = self.entity_extractor.extract_entities(item)

            # Calculate priority score
            priority = self._calculate_priority(item, entities)

            # Create processed event
            event = ProcessedNewsEvent(
                event_id=hashlib.md5(
                    f"{item.url}{item.published_at}".encode()
                ).hexdigest(),
                timestamp=item.published_at,
                source=item.source_name,
                headline=item.headline,
                summary=item.summary,
                url=item.url,
                entities=entities,
                priority_score=priority
            )

            processed_events.append(event)

        # Step 5: Sort by priority
        processed_events.sort(key=lambda x: x.priority_score, reverse=True)

        return processed_events

    def _calculate_priority(
        self,
        item: RawNewsItem,
        entities: ExtractedEntities
    ) -> float:
        """
        Calculate priority score for news event.

        Higher priority for:
        - Multiple commodities mentioned
        - Specific numerical metrics
        - Major events (OPEC meetings, government reports)
        - Reputable sources
        """
        score = 0.0

        # Commodity mentions (max 0.3)
        commodity_score = min(0.3, len(entities.commodities) * 0.1)
        score += commodity_score

        # Numerical metrics (max 0.3)
        metric_score = min(0.3, len(entities.metrics) * 0.1)
        score += metric_score

        # Major events (0.2)
        major_event_keywords = [
            "opec", "eia", "usda", "government", "federal",
            "production cut", "production increase", "forecast"
        ]
        text_lower = f"{item.headline} {item.summary}".lower()
        if any(kw in text_lower for kw in major_event_keywords):
            score += 0.2

        # Source credibility (max 0.2)
        credible_sources = [
            "reuters", "bloomberg", "wall street journal",
            "financial times", "eia", "usda"
        ]
        if any(source in item.source_name.lower() for source in credible_sources):
            score += 0.2

        return min(1.0, score)
```

## Common Pitfalls

**1. Missing Time-Critical News**
- **Problem**: Polling RSS feeds every hour misses breaking news
- **Why it happens**: RSS feeds are updated irregularly
- **Solution**: Use push notifications (webhooks), Twitter streaming API for real-time; poll critical sources every 1-5 minutes

**2. Duplicate Story Overload**
- **Problem**: Same Reuters story appears on 50 different sites
- **Why it happens**: News syndication spreads stories widely
- **Solution**: Content hashing + fuzzy headline matching; track "first seen" source

**3. False Relevance from Keywords**
- **Problem**: Article about "Gas prices at pump" tagged as natural gas news
- **Why it happens**: Keyword matching without context
- **Solution**: LLM-based relevance filtering; maintain commodity-specific jargon dictionary

**4. Ignoring Source Credibility**
- **Problem**: Treating random blog posts same as EIA reports
- **Why it happens**: All sources treated equally in initial processing
- **Solution**: Maintain source credibility scores; boost official sources (EIA, USDA, OPEC)

**5. Missing Implicit Information**
- **Problem**: "Saudi Arabia maintains current output" is actually news (expected cut didn't happen)
- **Why it happens**: Focusing only on explicit changes
- **Solution**: Track market expectations; LLM to identify "non-events" that are newsworthy

## Connections

**Builds on:**
- RSS/API fundamentals (data acquisition)
- Web scraping techniques (content extraction)
- Text preprocessing (cleaning, normalization)

**Leads to:**
- Sentiment extraction (analyzing news sentiment)
- Event detection (identifying market-moving events)
- Real-time alerting (notifying traders of critical news)

**Related to:**
- Natural language processing (entity recognition)
- Time-series analysis (news flow patterns)
- Information retrieval (relevance ranking)

## Practice Problems

1. **Multi-Source Aggregation**
   - Set up news acquisition for crude oil from:
     - 5 RSS feeds
     - NewsAPI
     - Twitter (key accounts like @EIAgov)
   - Run for 1 hour and analyze:
     - How many unique stories?
     - Average latency from publication to acquisition?
     - What percentage are duplicates?

2. **Relevance Classifier Evaluation**
   - Collect 100 commodity news articles
   - Manually label as relevant/not relevant to crude oil trading
   - Implement LLM-based relevance filter
   - Measure precision and recall
   - What confidence threshold optimizes F1 score?

3. **Entity Extraction Accuracy**
   - Extract entities from 20 commodity news articles
   - Validate extraction accuracy for:
     - Commodities mentioned
     - Companies mentioned
     - Numerical metrics (production, inventory)
   - Calculate accuracy for each entity type
   - What are common extraction errors?

4. **Real-Time Processing Pipeline**
   - Design a pipeline that processes news with <30 second latency
   - Handle 100 articles per minute
   - Constraints: API rate limits, deduplication storage
   - What architecture would you use? (Queue? Database? Cache?)

5. **Source Credibility Scoring**
   - Create a credibility scoring system for news sources
   - Factors to consider:
     - Official government sources (EIA, USDA) = highest
     - Major financial media (Bloomberg, Reuters)
     - Industry publications
     - Blogs and social media
   - Design a 0-1 scoring formula

## Further Reading

**News APIs and Data Sources:**
- NewsAPI Documentation - Multi-source news aggregation
- Twitter API v2 - Real-time tweet streaming
- Bloomberg Terminal API - Professional-grade financial news (subscription)

**Natural Language Processing:**
- spaCy Documentation - Named entity recognition
- "Fine-tuning BERT for Financial Entity Recognition" - Domain adaptation

**Real-Time Processing:**
- Apache Kafka Documentation - Message streaming for news pipelines
- Redis Documentation - Fast deduplication with sets/hashes

**Commodity News Sources:**
- Rigzone - Oil and gas industry news
- NaturalGasIntel - Natural gas markets
- Agriculture.com - Crop and livestock markets
- Kitco - Precious metals news

**Information Retrieval:**
- "Introduction to Information Retrieval" (Manning) - Relevance ranking fundamentals
- "Modern Information Retrieval" - Advanced filtering techniques
