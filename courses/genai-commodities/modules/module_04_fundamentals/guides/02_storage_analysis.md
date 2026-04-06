# Storage Analysis for Commodities with LLMs

> **Reading time:** ~13 min | **Module:** Module 4: Fundamentals | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Storage analysis uses LLM-augmented interpretation of inventory data to understand market tightness, seasonal patterns, and price implications by combining quantitative storage levels with qualitative context (weather forecasts, production disruptions, demand shifts) that determines whether curre...

</div>

## In Brief

Storage analysis uses LLM-augmented interpretation of inventory data to understand market tightness, seasonal patterns, and price implications by combining quantitative storage levels with qualitative context (weather forecasts, production disruptions, demand shifts) that determines whether current storage is adequate or concerning.

<div class="callout-insight">

**Insight:** Storage levels are meaningless without context—3,500 Bcf of natural gas storage is comfortable in October (injection season ending) but dangerously low in February (withdrawal season peak). LLMs excel at providing this contextual interpretation, understanding that "above 5-year average" can still be bullish if a polar vortex is forecasted or refineries are running at record rates.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of commodity storage like your phone battery:

**Simple view (wrong):**
- 50% battery = good
- 20% battery = concerning

**Context-dependent view (correct):**
- 50% battery at 9 AM with charger nearby = totally fine
- 50% battery at 9 PM with no charger, heavy usage ahead = concerning
- 20% battery at 10 PM at home = no problem
- 20% battery at midnight in unfamiliar city = very concerning

**Commodity storage example:**
- 400 million barrels crude storage in March = comfortable (refinery maintenance season, low demand)
- 400 million barrels in July = tight (driving season, high demand)
- 3,200 Bcf natural gas in November = below average but okay (injection season ended)
- 3,200 Bcf in January = dangerously low (peak heating demand)

The LLM understands these context dependencies that simple threshold rules miss.

## Formal Definition

Storage analysis is a function **SA: (S, C, T) → Assessment** where:

**Inputs:**
- **S** = storage metrics {absolute_level, change_from_prior, vs_5yr_avg, days_of_supply}
- **C** = context {season, weather_forecast, production_status, demand_outlook}
- **T** = time horizon {current, 1_month, 3_month, 6_month}

**Output:** Assessment = storage evaluation:
```
Assessment = {
  absolute_assessment: {
    level: value with unit,
    percentile: historical ranking (0-100),
    interpretation: "tight" | "comfortable" | "surplus"
  },
  relative_assessment: {
    vs_5yr_avg_pct: percentage deviation,
    vs_prior_week: change value,
    trend: "building" | "drawing" | "stable"
  },
  contextual_assessment: {
    seasonal_interpretation: "adequate" | "concerning" | "exceptional",
    days_of_supply: value,
    adequacy: "sufficient" | "marginal" | "insufficient",
    weather_adjusted: impact of forecast weather
  },
  price_implication: {
    direction: "bullish" | "bearish" | "neutral",
    confidence: [0, 1],
    reasoning: LLM-generated explanation,
    magnitude: "strong" | "moderate" | "weak"
  },
  risk_scenarios: {
    upside: "What could tighten storage (bullish)",
    downside: "What could add to storage (bearish)"
  }
}
```

**Key relationships:**
- **Low storage + High demand forecast** = Bullish (tight market)
- **High storage + Weak demand** = Bearish (oversupply)
- **Seasonal pattern deviation** = Signal of market imbalance

## Code Implementation

### Storage Data Analyzer


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from anthropic import Anthropic
import json

@dataclass
class StorageMetrics:
    """Current storage metrics."""
    commodity: str
    absolute_level: float
    unit: str
    reporting_date: datetime

    # Relative metrics
    prior_week_level: float
    weekly_change: float
    five_year_avg: float
    five_year_min: float
    five_year_max: float

    # Calculated
    vs_5yr_avg_pct: float
    percentile_rank: float  # 0-100, where in historical distribution

@dataclass
class StorageAssessment:
    """Complete storage assessment."""
    metrics: StorageMetrics
    absolute_interpretation: str  # "tight", "comfortable", "surplus"
    seasonal_interpretation: str  # "adequate", "concerning", "exceptional"
    days_of_supply: float
    price_implication: str  # "bullish", "bearish", "neutral"
    confidence: float
    narrative: str
    risk_scenarios: Dict[str, str]

class StorageAnalyzer:
    """
    Analyze commodity storage with LLM augmentation.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def analyze_crude_storage(
        self,
        current_storage: float,
        historical_data: pd.DataFrame,
        refinery_utilization: float,
        news_context: Optional[List[str]] = None
    ) -> StorageAssessment:
        """
        Analyze crude oil storage situation.

        Args:
            current_storage: Current inventory level (million barrels)
            historical_data: DataFrame with historical storage levels
            refinery_utilization: Current refinery capacity utilization (%)
            news_context: Recent relevant news

        Returns:
            Complete storage assessment
        """
        # Calculate metrics
        metrics = self._calculate_storage_metrics(
            commodity="crude_oil",
            current_storage=current_storage,
            historical_data=historical_data
        )

        # Calculate days of supply
        # Assume typical refinery inputs of ~16 million bpd
        daily_demand = refinery_utilization / 100 * 17  # million bpd at 100% util
        days_of_supply = current_storage / daily_demand

        # Get seasonal context
        month = datetime.now().month
        seasonal_context = self._get_crude_seasonal_context(month)

        # LLM interpretation
        interpretation = self._llm_interpret_storage(
            metrics,
            days_of_supply,
            seasonal_context,
            {
                "refinery_utilization": refinery_utilization,
                "news": news_context or []
            }
        )

        return StorageAssessment(
            metrics=metrics,
            absolute_interpretation=interpretation['absolute'],
            seasonal_interpretation=interpretation['seasonal'],
            days_of_supply=days_of_supply,
            price_implication=interpretation['price_implication'],
            confidence=interpretation['confidence'],
            narrative=interpretation['narrative'],
            risk_scenarios=interpretation['risk_scenarios']
        )

    def analyze_natural_gas_storage(
        self,
        current_storage: float,
        historical_data: pd.DataFrame,
        weather_forecast: Dict[str, any]
    ) -> StorageAssessment:
        """
        Analyze natural gas storage (highly seasonal).

        Natural gas storage interpretation is heavily dependent on:
        - Season (injection vs withdrawal)
        - Weather forecast (heating/cooling demand)
        - Production trends
        """
        metrics = self._calculate_storage_metrics(
            commodity="natural_gas",
            current_storage=current_storage,
            historical_data=historical_data
        )

        # Seasonal context for natural gas
        month = datetime.now().month
        if month in [11, 12, 1, 2, 3]:
            season = "withdrawal_season"
            typical_weekly_change = -100  # Bcf per week
        elif month in [4, 5, 6, 7, 8, 9, 10]:
            season = "injection_season"
            typical_weekly_change = 80  # Bcf per week
        else:
            season = "shoulder"
            typical_weekly_change = 0

        # Calculate days of supply (more complex for nat gas)
        # Typical winter demand: ~140 Bcf/day, summer: ~80 Bcf/day
        if season == "withdrawal_season":
            daily_demand = 140
        else:
            daily_demand = 80

        days_of_supply = current_storage / daily_demand

        # LLM interpretation with weather
        interpretation = self._llm_interpret_gas_storage(
            metrics,
            season,
            weather_forecast,
            days_of_supply
        )

        return StorageAssessment(
            metrics=metrics,
            absolute_interpretation=interpretation['absolute'],
            seasonal_interpretation=interpretation['seasonal'],
            days_of_supply=days_of_supply,
            price_implication=interpretation['price_implication'],
            confidence=interpretation['confidence'],
            narrative=interpretation['narrative'],
            risk_scenarios=interpretation['risk_scenarios']
        )

    def _calculate_storage_metrics(
        self,
        commodity: str,
        current_storage: float,
        historical_data: pd.DataFrame
    ) -> StorageMetrics:
        """
        Calculate storage metrics from current and historical data.
        """
        # Get prior week
        prior_week = historical_data.iloc[-2]['storage'] if len(historical_data) > 1 else current_storage
        weekly_change = current_storage - prior_week

        # Calculate 5-year average for same week
        current_week = datetime.now().isocalendar()[1]

        # Filter to same week of year over past 5 years
        five_year_data = historical_data[
            historical_data.index.isocalendar().week == current_week
        ].tail(5)

        five_year_avg = five_year_data['storage'].mean()
        five_year_min = five_year_data['storage'].min()
        five_year_max = five_year_data['storage'].max()

        # Calculate percentile rank in historical distribution
        all_historical = historical_data['storage']
        percentile_rank = (all_historical < current_storage).sum() / len(all_historical) * 100

        # Vs 5-year average
        vs_5yr_avg_pct = (current_storage - five_year_avg) / five_year_avg * 100

        return StorageMetrics(
            commodity=commodity,
            absolute_level=current_storage,
            unit="million_barrels" if commodity == "crude_oil" else "Bcf",
            reporting_date=datetime.now(),
            prior_week_level=prior_week,
            weekly_change=weekly_change,
            five_year_avg=five_year_avg,
            five_year_min=five_year_min,
            five_year_max=five_year_max,
            vs_5yr_avg_pct=vs_5yr_avg_pct,
            percentile_rank=percentile_rank
        )

    def _get_crude_seasonal_context(self, month: int) -> str:
        """
        Get seasonal context for crude oil.
        """
        if month in [5, 6, 7, 8, 9]:
            return "driving_season"  # High gasoline demand
        elif month in [11, 12, 1, 2]:
            return "winter"  # High distillate demand
        elif month in [3, 4]:
            return "refinery_maintenance"  # Lower crude demand
        else:
            return "shoulder"

    def _llm_interpret_storage(
        self,
        metrics: StorageMetrics,
        days_of_supply: float,
        seasonal_context: str,
        additional_context: Dict
    ) -> Dict[str, any]:
        """
        Use LLM to interpret storage situation.
        """
        news_context = "\n".join(additional_context.get('news', [])[:3])

        prompt = f"""Analyze this crude oil storage situation.

STORAGE METRICS:
- Current Level: {metrics.absolute_level:.1f} {metrics.unit}
- Prior Week: {metrics.prior_week_level:.1f} {metrics.unit}
- Weekly Change: {metrics.weekly_change:+.1f} {metrics.unit}
- 5-Year Average (this week): {metrics.five_year_avg:.1f} {metrics.unit}
- Vs 5-Year Average: {metrics.vs_5yr_avg_pct:+.1f}%
- Historical Percentile: {metrics.percentile_rank:.0f}th percentile

CONTEXT:
- Days of Supply: {days_of_supply:.1f} days
- Season: {seasonal_context}
- Refinery Utilization: {additional_context.get('refinery_utilization', 'N/A')}%

RECENT NEWS:
{news_context if news_context else 'No recent news'}

Provide comprehensive storage assessment as JSON:
{{
  "absolute": "tight | comfortable | surplus",
  "seasonal": "adequate | concerning | exceptional",
  "price_implication": "bullish | bearish | neutral",
  "magnitude": "strong | moderate | weak",
  "confidence": 0.0-1.0,
  "narrative": "2-3 paragraph analysis explaining storage situation",
  "key_factors": [
    "Factor 1: Why storage is at this level",
    "Factor 2: What this means for markets",
    "Factor 3: How this compares historically"
  ],
  "risk_scenarios": {{
    "upside": "What could tighten storage further (bullish scenario)",
    "downside": "What could add to storage (bearish scenario)"
  }}
}}

INTERPRETATION GUIDELINES:
- Storage above 5-year average = Generally bearish (oversupply)
- Storage below 5-year average = Generally bullish (tight market)
- But consider:
  - Driving season (summer): Need more storage
  - Refinery maintenance: Can tolerate lower storage
  - Days of supply: <25 days = concerning, >35 days = comfortable
  - Weekly change direction: Building vs drawing
  - Refinery utilization: High util = bullish (strong demand)"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)

    def _llm_interpret_gas_storage(
        self,
        metrics: StorageMetrics,
        season: str,
        weather_forecast: Dict[str, any],
        days_of_supply: float
    ) -> Dict[str, any]:
        """
        Interpret natural gas storage with seasonal and weather context.
        """
        prompt = f"""Analyze this natural gas storage situation.

STORAGE METRICS:
- Current Level: {metrics.absolute_level:.0f} Bcf
- Prior Week: {metrics.prior_week_level:.0f} Bcf
- Weekly Change: {metrics.weekly_change:+.0f} Bcf
- 5-Year Average (this week): {metrics.five_year_avg:.0f} Bcf
- Vs 5-Year Average: {metrics.vs_5yr_avg_pct:+.1f}%
- Historical Percentile: {metrics.percentile_rank:.0f}th percentile

CONTEXT:
- Season: {season}
- Days of Supply: {days_of_supply:.1f} days
- Weather Forecast: {json.dumps(weather_forecast, indent=2)}

Provide storage assessment as JSON:
{{
  "absolute": "tight | comfortable | surplus",
  "seasonal": "adequate | concerning | exceptional",
  "price_implication": "bullish | bearish | neutral",
  "magnitude": "strong | moderate | weak",
  "confidence": 0.0-1.0,
  "narrative": "Comprehensive analysis incorporating weather forecast",
  "risk_scenarios": {{
    "upside": "Bullish scenarios (e.g., colder than forecast winter)",
    "downside": "Bearish scenarios (e.g., warmer than forecast, high production)"
  }}
}}

CRITICAL SEASONAL CONTEXT:
- INJECTION SEASON (Apr-Oct): Storage should be building
  - Below 5-year average = Bullish (can't build enough for winter)
  - Above 5-year average = Bearish (ample supply)

- WITHDRAWAL SEASON (Nov-Mar): Storage should be drawing
  - Below 5-year average = Very bullish (might run short)
  - Above 5-year average = Bearish (oversupply)
  - Weather forecast is CRITICAL - cold weather exponentially increases draws

- Consider:
  - Typical withdrawal season peak demand: ~140 Bcf/day
  - Typical injection season demand: ~80 Bcf/day
  - Storage capacity: ~4,100 Bcf maximum
  - Days of supply: <25 days = critically tight, >40 days = comfortable"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)
```

</div>
</div>

### Historical Pattern Comparison


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">storagepatternanalyzer.py</span>

```python
class StoragePatternAnalyzer:
    """
    Analyze storage patterns over time.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def compare_to_historical_patterns(
        self,
        current_metrics: StorageMetrics,
        historical_data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Compare current storage to historical patterns.
        """
        # Find similar historical periods
        similar_periods = self._find_similar_periods(
            current_metrics,
            historical_data
        )

        # Analyze what happened after those periods
        outcomes = self._analyze_outcomes(similar_periods, historical_data)

        # LLM synthesizes pattern analysis
        pattern_analysis = self._llm_pattern_synthesis(
            current_metrics,
            similar_periods,
            outcomes
        )

        return pattern_analysis

    def _find_similar_periods(
        self,
        current_metrics: StorageMetrics,
        historical_data: pd.DataFrame,
        tolerance: float = 5.0
    ) -> List[Dict]:
        """
        Find historical periods with similar storage characteristics.

        Args:
            tolerance: Percentage tolerance for similarity (5% default)
        """
        similar = []

        # Get current week of year
        current_week = current_metrics.reporting_date.isocalendar()[1]

        # Filter to same week across years
        same_week_data = historical_data[
            historical_data.index.isocalendar().week == current_week
        ]

        for date, row in same_week_data.iterrows():
            # Check if storage level is similar
            pct_diff = abs(row['storage'] - current_metrics.absolute_level) / current_metrics.absolute_level * 100

            if pct_diff <= tolerance:
                similar.append({
                    'date': date,
                    'storage': row['storage'],
                    'pct_diff': pct_diff
                })

        return similar

    def _analyze_outcomes(
        self,
        similar_periods: List[Dict],
        historical_data: pd.DataFrame,
        forward_weeks: int = 12
    ) -> List[Dict]:
        """
        Analyze what happened in the weeks following similar periods.
        """
        outcomes = []

        for period in similar_periods:
            start_date = period['date']
            end_date = start_date + timedelta(weeks=forward_weeks)

            # Get forward data
            forward_data = historical_data[
                (historical_data.index > start_date) &
                (historical_data.index <= end_date)
            ]

            if len(forward_data) > 0:
                outcomes.append({
                    'period': start_date,
                    'initial_storage': period['storage'],
                    'final_storage': forward_data.iloc[-1]['storage'],
                    'change': forward_data.iloc[-1]['storage'] - period['storage'],
                    'max_storage': forward_data['storage'].max(),
                    'min_storage': forward_data['storage'].min()
                })

        return outcomes

    def _llm_pattern_synthesis(
        self,
        current_metrics: StorageMetrics,
        similar_periods: List[Dict],
        outcomes: List[Dict]
    ) -> Dict[str, any]:
        """
        Synthesize pattern analysis with LLM.
        """
        # Format historical comparisons
        comparisons = "\n".join([
            f"- {outcome['period'].strftime('%Y-%m-%d')}: Started at {outcome['initial_storage']:.0f}, "
            f"ended at {outcome['final_storage']:.0f} ({outcome['change']:+.0f} change)"
            for outcome in outcomes[:5]
        ])

        prompt = f"""Analyze storage patterns based on historical comparisons.

CURRENT SITUATION:
- Storage: {current_metrics.absolute_level:.0f} {current_metrics.unit}
- Vs 5-Year Average: {current_metrics.vs_5yr_avg_pct:+.1f}%

SIMILAR HISTORICAL PERIODS (same week of year, similar storage level):
{comparisons}

Based on these historical patterns, provide analysis as JSON:
{{
  "typical_outcome": "Description of what typically happened",
  "storage_forecast": {{
    "4_week": value,
    "12_week": value,
    "reasoning": "Why this forecast based on patterns"
  }},
  "pattern_strength": "How reliable are these patterns (0.0-1.0)",
  "key_differences": "How current situation differs from historical periods",
  "confidence": 0.0-1.0
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)
```

</div>
</div>

## Common Pitfalls

**1. Comparing Absolute Levels Across Seasons**
- **Problem**: "3,200 Bcf natural gas storage is good" without checking the month
- **Why it happens**: Not accounting for seasonal patterns
- **Solution**: Always compare to same period in prior years; use seasonal adjustments

**2. Ignoring Rate of Change**
- **Problem**: Focusing only on current level, not whether storage is building or drawing
- **Why it happens**: Missing the dynamic aspect of storage
- **Solution**: Track weekly changes; compare actual change to typical seasonal pattern

**3. No Weather Adjustment for Natural Gas**
- **Problem**: Saying storage is "comfortable" without checking weather forecast
- **Why it happens**: Treating natural gas like crude oil (less weather-sensitive)
- **Solution**: Always incorporate weather forecasts for natural gas analysis

**4. Crude Storage Without Refinery Context**
- **Problem**: High crude storage interpreted as bearish during refinery maintenance season
- **Why it happens**: Not considering demand side (refineries)
- **Solution**: Check refinery utilization rates; maintenance season = lower demand is normal

**5. Percentile Rank Misinterpretation**
- **Problem**: "80th percentile storage = surplus" when it might be appropriate for the season
- **Why it happens**: Using absolute percentile without seasonal context
- **Solution**: Calculate percentile within same season/month of prior years

## Connections

**Builds on:**
- Data acquisition (getting storage reports)
- Time-series analysis (identifying trends and patterns)
- Statistical comparison (percentiles, z-scores)

**Leads to:**
- Price forecasting (storage levels predict price moves)
- Trading signals (tight storage = potential long signal)
- Risk management (low storage = supply risk)

**Related to:**
- Supply/demand balancing (storage is the buffer)
- Seasonal analysis (understanding annual patterns)
- Weather modeling (for natural gas and agriculture)

## Practice Problems

1. **Crude Oil Storage Assessment**
   - Current: 425 million barrels
   - 5-year average: 450 million barrels
   - Season: Late May (driving season approaching)
   - Refinery utilization: 94% (very high)
   - Assess: Bullish or bearish? Why?

2. **Natural Gas Seasonal Context**
   - Current storage: 3,100 Bcf in early November
   - 5-year average: 3,400 Bcf
   - Weather forecast: Normal temperatures
   - Is this concerning? Calculate days of supply (winter demand ~140 Bcf/day)

3. **Rate of Change Analysis**
   - Week 1: 3,500 Bcf
   - Week 2: 3,450 Bcf (draw of 50 Bcf)
   - Week 3: 3,370 Bcf (draw of 80 Bcf)
   - Week 4: 3,250 Bcf (draw of 120 Bcf)
   - What's the trend? Price implication?

4. **Historical Pattern Matching**
   - Current: 440 million barrels crude in June
   - Similar periods: June 2019 (438 MMB), June 2020 (520 MMB), June 2021 (455 MMB)
   - Those periods saw: -15 MMB, +10 MMB, -8 MMB changes over next month
   - Forecast next month's change

5. **LLM Prompt Design**
   - Design a prompt for LLM to assess agricultural commodity storage (corn)
   - Must incorporate: Harvest timing, export demand, ethanol production
   - What additional context would you provide?

<div class="callout-insight">

**Insight:** Understanding storage analysis for commodities with llms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Crude Oil Storage:**
- EIA: "Petroleum Stocks and Days of Supply" - Official methodology
- "Understanding Cushing Storage" - Key storage hub analysis

**Natural Gas Storage:**
- EIA: "Natural Gas Weekly Update" - Storage interpretation guide
- "Weather Impacts on Natural Gas Demand" - Temperature-demand relationships

**Agricultural Storage:**
- USDA: "Grain Stocks Reports" - Inventory data and interpretation
- "Stocks-to-Use Ratios" - Key metric for grain markets

**Statistical Analysis:**
- "Seasonal Decomposition of Time Series" - Separating trend from seasonality
- "Percentile Analysis in Commodity Markets" - Ranking current conditions

**LLM Applications:**
- "Contextual Analysis with Language Models" - Incorporating qualitative factors
- "Prompt Engineering for Market Analysis" - Designing effective prompts

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_storage_analysis_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_crude_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_balance_modeling.md">
  <div class="link-card-title">01 Balance Modeling</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_supply_demand.md">
  <div class="link-card-title">01 Supply Demand</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

