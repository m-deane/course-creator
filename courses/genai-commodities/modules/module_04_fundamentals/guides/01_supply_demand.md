# Supply/Demand Modeling with LLMs

> **Reading time:** ~10 min | **Module:** Module 4: Fundamentals | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Supply/demand modeling with LLMs combines traditional fundamental analysis (production data, inventory levels, consumption patterns) with LLM-powered synthesis of unstructured information (geopolitical events, weather impacts, policy changes) to build comprehensive supply/demand balances that inf...

</div>

## In Brief

Supply/demand modeling with LLMs combines traditional fundamental analysis (production data, inventory levels, consumption patterns) with LLM-powered synthesis of unstructured information (geopolitical events, weather impacts, policy changes) to build comprehensive supply/demand balances that inform commodity price forecasts.

<div class="callout-insight">

**Insight:** Traditional supply/demand models rely on structured data (EIA inventories, USDA production estimates) but miss critical context from unstructured sources—a surprise OPEC announcement, drought in the Midwest, or refinery outage. LLMs excel at extracting and reasoning over this unstructured information, allowing quantitative models to incorporate qualitative factors that actually move markets.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of supply/demand modeling like forecasting your home heating costs:

**Traditional approach:**
- Look at last year's gas usage
- Apply inflation adjustment
- Done

**Problems:**
- Misses that you added insulation (less demand)
- Misses polar vortex forecast (more demand)
- Misses pipeline construction (more supply)

**LLM-augmented approach:**
- Get historical usage data (structured)
- LLM reads news: "New pipeline increases regional supply 15%"
- LLM processes weather forecast: "Colder than normal winter expected"
- LLM notes: "Home insulation project completed last month"
- Combines all factors: Higher demand (cold) + Lower home usage (insulation) + More supply (pipeline) = Net balanced, prices stable

The LLM bridges the gap between numbers and narrative, creating a complete picture.

## Formal Definition

An LLM-augmented supply/demand model is a system **SDM: (D_structured, D_unstructured) → Balance** where:

**Inputs:**
- **D_structured** = quantitative data {production, inventory, consumption, imports, exports}
- **D_unstructured** = qualitative information {news, forecasts, analyst commentary, policy docs}

**Output:** Balance = supply/demand assessment:
```
Balance = {
  supply: {
    current_level: value with unit,
    trend: increasing | decreasing | stable,
    forecast: {1_month, 3_month, 6_month},
    key_factors: [production, imports, inventory_release],
    risks: [upside_scenarios, downside_scenarios]
  },
  demand: {
    current_level: value with unit,
    trend: increasing | decreasing | stable,
    forecast: {1_month, 3_month, 6_month},
    key_factors: [consumption, exports, seasonal],
    risks: [upside_scenarios, downside_scenarios]
  },
  balance: {
    net_position: surplus | deficit | balanced,
    magnitude: value with unit,
    price_implication: bullish | bearish | neutral,
    confidence: [0, 1],
    narrative: LLM-generated explanation
  }
}
```

**Key equation:**
```
Balance = Supply - Demand
Supply = Production + Imports + Inventory_Release
Demand = Consumption + Exports + Inventory_Build
```

**LLM augmentation adds:**
- Contextual interpretation of numbers
- Integration of qualitative factors
- Scenario analysis
- Natural language explanation

## Code Implementation

### Supply/Demand Data Aggregation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from anthropic import Anthropic
import json

@dataclass
class SupplyComponent:
    """Individual supply source."""
    source: str  # "domestic_production", "imports", "inventory_release"
    value: float
    unit: str
    time_period: str
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float

@dataclass
class DemandComponent:
    """Individual demand source."""
    source: str  # "domestic_consumption", "exports", "inventory_build"
    value: float
    unit: str
    time_period: str
    trend: str
    confidence: float

@dataclass
class SupplyDemandBalance:
    """Complete supply/demand balance."""
    commodity: str
    reporting_date: datetime

    # Supply side
    total_supply: float
    supply_components: List[SupplyComponent]
    supply_trend: str

    # Demand side
    total_demand: float
    demand_components: List[DemandComponent]
    demand_trend: str

    # Balance
    net_balance: float  # Supply - Demand
    balance_status: str  # "surplus", "deficit", "balanced"
    price_implication: str  # "bullish", "bearish", "neutral"

    # Forecasts
    one_month_forecast: Optional[float]
    three_month_forecast: Optional[float]

    # Narrative
    narrative: str
    key_factors: List[str]
    risks: Dict[str, List[str]]

class SupplyDemandAnalyzer:
    """
    Analyze supply/demand using LLM augmentation.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def build_crude_oil_balance(
        self,
        eia_data: pd.DataFrame,
        news_context: List[str],
        forecast_horizon: str = "3_month"
    ) -> SupplyDemandBalance:
        """
        Build crude oil supply/demand balance.

        Args:
            eia_data: DataFrame with EIA data (production, imports, etc.)
            news_context: Recent news articles about oil markets
            forecast_horizon: "1_month", "3_month", "6_month"

        Returns:
            Complete supply/demand balance
        """
        # Step 1: Extract structured data
        latest_data = eia_data.iloc[-1]

        # Supply components (million barrels per day)
        supply_components = [
            SupplyComponent(
                source="domestic_production",
                value=latest_data['production'],
                unit="million_bpd",
                time_period="weekly",
                trend=self._calculate_trend(eia_data['production']),
                confidence=0.95  # EIA data is high quality
            ),
            SupplyComponent(
                source="imports",
                value=latest_data['imports'],
                unit="million_bpd",
                time_period="weekly",
                trend=self._calculate_trend(eia_data['imports']),
                confidence=0.90
            )
        ]

        total_supply = sum(c.value for c in supply_components)

        # Demand components
        demand_components = [
            DemandComponent(
                source="refinery_inputs",
                value=latest_data['refinery_inputs'],
                unit="million_bpd",
                time_period="weekly",
                trend=self._calculate_trend(eia_data['refinery_inputs']),
                confidence=0.95
            ),
            DemandComponent(
                source="exports",
                value=latest_data['exports'],
                unit="million_bpd",
                time_period="weekly",
                trend=self._calculate_trend(eia_data['exports']),
                confidence=0.90
            )
        ]

        total_demand = sum(c.value for c in demand_components)

        # Step 2: Use LLM to interpret data + news context
        interpretation = self._llm_interpret_balance(
            supply_components,
            demand_components,
            news_context,
            latest_data
        )

        # Step 3: Generate forecast using LLM + structured data
        forecast = self._llm_forecast_balance(
            eia_data,
            news_context,
            forecast_horizon
        )

        # Step 4: Build complete balance
        net_balance = total_supply - total_demand

        if net_balance > 0.5:
            balance_status = "surplus"
            price_implication = "bearish"
        elif net_balance < -0.5:
            balance_status = "deficit"
            price_implication = "bullish"
        else:
            balance_status = "balanced"
            price_implication = "neutral"

        return SupplyDemandBalance(
            commodity="crude_oil",
            reporting_date=datetime.now(),
            total_supply=total_supply,
            supply_components=supply_components,
            supply_trend=interpretation['supply_trend'],
            total_demand=total_demand,
            demand_components=demand_components,
            demand_trend=interpretation['demand_trend'],
            net_balance=net_balance,
            balance_status=balance_status,
            price_implication=price_implication,
            one_month_forecast=forecast.get('1_month'),
            three_month_forecast=forecast.get('3_month'),
            narrative=interpretation['narrative'],
            key_factors=interpretation['key_factors'],
            risks=interpretation['risks']
        )

    def _calculate_trend(self, series: pd.Series, periods: int = 4) -> str:
        """
        Calculate trend direction from time series.
        """
        recent = series.tail(periods)

        if len(recent) < 2:
            return "stable"

        # Simple linear trend
        slope = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        pct_change = slope / recent.mean() * 100

        if pct_change > 2:
            return "increasing"
        elif pct_change < -2:
            return "decreasing"
        else:
            return "stable"

    def _llm_interpret_balance(
        self,
        supply_components: List[SupplyComponent],
        demand_components: List[DemandComponent],
        news_context: List[str],
        latest_data: pd.Series
    ) -> Dict[str, any]:
        """
        Use LLM to interpret supply/demand balance with context.
        """
        # Format supply/demand data
        supply_summary = "\n".join([
            f"- {c.source}: {c.value:.2f} {c.unit} ({c.trend})"
            for c in supply_components
        ])

        demand_summary = "\n".join([
            f"- {c.source}: {c.value:.2f} {c.unit} ({c.trend})"
            for c in demand_components
        ])

        # Format news context
        news_summary = "\n\n".join(news_context[:5])  # Top 5 recent news

        prompt = f"""Analyze this crude oil supply/demand balance.

CURRENT SUPPLY:
{supply_summary}
Total Supply: {sum(c.value for c in supply_components):.2f} million bpd

CURRENT DEMAND:
{demand_summary}
Total Demand: {sum(c.value for c in demand_components):.2f} million bpd

INVENTORY LEVEL: {latest_data.get('inventory', 'N/A')} million barrels

RECENT NEWS CONTEXT:
{news_summary}

Provide comprehensive analysis as JSON:
{{
  "supply_trend": "increasing | decreasing | stable",
  "demand_trend": "increasing | decreasing | stable",
  "narrative": "2-3 paragraph analysis explaining the balance, incorporating news context",
  "key_factors": [
    "List 3-5 key factors driving supply/demand",
    "Example: OPEC production cuts limiting supply",
    "Example: Strong refinery demand from high crack spreads"
  ],
  "risks": {{
    "upside": ["Factors that could tighten supply/demand", "..."],
    "downside": ["Factors that could loosen supply/demand", "..."]
  }},
  "price_implication_reasoning": "Explain why this balance is bullish/bearish/neutral for prices"
}}

Consider:
- How news context affects fundamentals
- Seasonal factors (if mentioned)
- Geopolitical impacts
- Supply/demand trajectory (not just current level)"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)

    def _llm_forecast_balance(
        self,
        historical_data: pd.DataFrame,
        news_context: List[str],
        forecast_horizon: str
    ) -> Dict[str, float]:
        """
        Generate supply/demand forecast using LLM.
        """
        # Prepare historical context
        recent_supply = historical_data['production'].tail(12).tolist()
        recent_demand = historical_data['refinery_inputs'].tail(12).tolist()

        news_summary = "\n".join(news_context[:3])

        prompt = f"""Forecast crude oil supply/demand balance.

HISTORICAL SUPPLY (last 12 weeks, million bpd):
{recent_supply}

HISTORICAL DEMAND (last 12 weeks, million bpd):
{recent_demand}

CURRENT LEVEL:
Supply: {recent_supply[-1]:.2f} million bpd
Demand: {recent_demand[-1]:.2f} million bpd
Balance: {recent_supply[-1] - recent_demand[-1]:.2f} million bpd

CONTEXT:
{news_summary}

Provide forecasts as JSON:
{{
  "1_month": {{
    "supply": value,
    "demand": value,
    "balance": value,
    "reasoning": "Why these levels expected"
  }},
  "3_month": {{
    "supply": value,
    "demand": value,
    "balance": value,
    "reasoning": "Why these levels expected"
  }},
  "confidence": 0.0-1.0,
  "key_assumptions": ["Assumption 1", "Assumption 2"]
}}

Consider:
- Current trends (are they accelerating/decelerating?)
- Known future events (OPEC meetings, refinery maintenance)
- Seasonal patterns
- Policy changes
- Economic outlook"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1536,
            messages=[{"role": "user", "content": prompt}]
        )

        forecast_data = json.loads(response.content[0].text)

        return {
            '1_month': forecast_data['1_month']['balance'],
            '3_month': forecast_data['3_month']['balance']
        }
```

</div>

### Natural Gas Supply/Demand (Seasonal Model)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">naturalgassdanalyzer.py</span>
</div>

```python
class NaturalGasSDAnalyzer:
    """
    Natural gas supply/demand with seasonal awareness.

    Natural gas has strong seasonality:
    - Winter: High demand (heating)
    - Summer: High demand (cooling, power generation)
    - Spring/Fall: Low demand (shoulder seasons)
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def build_seasonal_balance(
        self,
        eia_storage_data: pd.DataFrame,
        weather_forecast: Dict[str, any],
        current_month: int
    ) -> SupplyDemandBalance:
        """
        Build natural gas balance with seasonal adjustments.
        """
        # Determine season
        if current_month in [12, 1, 2]:
            season = "winter"
            seasonal_demand_factor = 1.3  # 30% higher demand
        elif current_month in [6, 7, 8]:
            season = "summer"
            seasonal_demand_factor = 1.15  # 15% higher demand
        else:
            season = "shoulder"
            seasonal_demand_factor = 0.9  # 10% lower demand

        # Get storage levels
        current_storage = eia_storage_data.iloc[-1]['storage']
        five_year_avg = eia_storage_data['storage_5yr_avg'].iloc[-1]

        storage_vs_avg_pct = (current_storage - five_year_avg) / five_year_avg * 100

        # LLM interprets storage level with seasonal context
        interpretation = self._interpret_storage_seasonally(
            current_storage,
            five_year_avg,
            season,
            weather_forecast
        )

        # Build balance (simplified for natural gas)
        return interpretation

    def _interpret_storage_seasonally(
        self,
        current_storage: float,
        five_year_avg: float,
        season: str,
        weather_forecast: Dict
    ) -> Dict[str, any]:
        """
        Interpret storage level considering season and weather.
        """
        prompt = f"""Analyze natural gas storage situation.

STORAGE:
- Current: {current_storage:.0f} Bcf
- 5-Year Average: {five_year_avg:.0f} Bcf
- Vs Average: {((current_storage - five_year_avg) / five_year_avg * 100):+.1f}%

SEASON: {season}

WEATHER FORECAST:
{json.dumps(weather_forecast, indent=2)}

Provide analysis as JSON:
{{
  "storage_assessment": "tight | comfortable | surplus",
  "price_implication": "bullish | bearish | neutral",
  "reasoning": "Explain considering season and weather",
  "risk_scenarios": {{
    "bullish": "What could tighten supply (e.g., colder than forecast)",
    "bearish": "What could loosen supply (e.g., warmer than forecast)"
  }}
}}

IMPORTANT - Seasonal Context:
- Winter: Low storage is very bullish (heating demand peak)
- Summer: Moderate storage is neutral (power gen demand)
- Shoulder: High storage is bearish (refilling season)

Storage above 5-year average is generally bearish.
Storage below 5-year average is generally bullish.
But magnitude depends on season."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)
```

</div>

## Common Pitfalls

**1. Ignoring Quality of Data Sources**
- **Problem**: Treating preliminary estimates same as final data
- **Why it happens**: All numbers look equally authoritative
- **Solution**: Weight data by source quality; EIA > industry estimates > news reports

**2. Missing Seasonal Adjustments**
- **Problem**: Comparing winter demand to summer demand without adjustment
- **Why it happens**: Not normalizing for seasonal patterns
- **Solution**: Use seasonal adjustments; compare to same period in prior years

**3. Static Supply/Demand Models**
- **Problem**: Using last week's balance to forecast next month
- **Why it happens**: Not incorporating forward-looking information
- **Solution**: LLM extracts forecasts and expected changes from news/reports

**4. Ignoring Inventory Levels**
- **Problem**: Focusing only on flow (production/consumption), not stock (inventory)
- **Why it happens**: Inventory is a buffer that complicates simple supply/demand
- **Solution**: Include inventory changes as supply/demand component

**5. No Uncertainty Quantification**
- **Problem**: Presenting single-point forecasts without confidence intervals
- **Why it happens**: Overconfidence in model accuracy
- **Solution**: Generate scenarios (base case, upside, downside); report confidence levels

## Connections

**Builds on:**
- Data processing (acquiring and cleaning supply/demand data)
- RAG systems (retrieving relevant context)
- Time-series analysis (identifying trends)

**Leads to:**
- Price forecasting (using supply/demand to predict prices)
- Trading strategies (positioning based on balances)
- Risk management (scenario analysis)

**Related to:**
- Macroeconomic analysis (GDP growth affects demand)
- Weather modeling (temperature impacts demand)
- Geopolitical analysis (conflicts affect supply)

## Practice Problems

1. **Crude Oil Balance**
   - Production: 13.0 million bpd (increasing)
   - Imports: 6.5 million bpd (stable)
   - Refinery inputs: 16.8 million bpd (increasing)
   - Exports: 3.2 million bpd (increasing)
   - Calculate: Total supply, Total demand, Net balance
   - What's the price implication?

2. **Seasonal Adjustment**
   - Natural gas storage: 3,200 Bcf in November
   - 5-year average for November: 3,500 Bcf
   - Weather forecast: Colder than normal December
   - Assess: Is this bullish or bearish? Why?

3. **Forecast Integration**
   - Current crude production: 12.8 million bpd
   - News: "OPEC announces 1 million bpd cut starting next month"
   - News: "U.S. shale producers increase rig count"
   - Forecast production 3 months out (justify your reasoning)

4. **Multi-Component Balance**
   - For corn:
     - Production: 15 billion bushels (good harvest)
     - Domestic use: 11 billion bushels
     - Exports: 2.5 billion bushels
     - Carryover from last year: 1.5 billion bushels
   - Calculate ending stocks
   - Is this a tight or comfortable balance?

5. **LLM-Augmented Scenario**
   - You have EIA data showing crude inventory at 430 million barrels
   - News mentions: "Hurricane threatens Gulf Coast refineries"
   - How would you use an LLM to assess supply/demand impact?
   - What prompt would you design?

<div class="callout-insight">

**Insight:** Understanding supply/demand modeling with llms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Commodity Fundamentals:**
- "Commodity Markets and Prices" - Supply/demand theory
- EIA: "Energy Supply and Demand Modeling" - Official methodology

**Natural Gas Specific:**
- "Fundamentals of Natural Gas Markets" - Seasonal patterns
- EIA: "Natural Gas Storage Explained" - Inventory dynamics

**Agricultural Commodities:**
- USDA: "WASDE Methodology" - Supply/demand estimates
- "Agricultural Commodity Markets and Trade" - Market structure

**LLM Applications:**
- "Large Language Models for Economic Forecasting" (2024)
- "Augmenting Quantitative Models with Qualitative Information"

**Production Systems:**
- "Building Fundamental Analysis Dashboards"
- "Real-Time Supply/Demand Monitoring Systems"

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="../notebooks/01_crude_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_balance_modeling.md">
  <div class="link-card-title">01 Balance Modeling</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_storage_analysis.md">
  <div class="link-card-title">02 Storage Analysis</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

