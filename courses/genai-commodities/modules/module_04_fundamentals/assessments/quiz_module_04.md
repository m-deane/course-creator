# Quiz: Module 4 - Fundamentals Modeling with LLMs

**Course:** GenAI for Commodity Markets
**Module:** 4 - Fundamentals Modeling
**Total Points:** 100
**Time Limit:** 30 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your ability to build supply/demand models enhanced by LLMs, integrate qualitative and quantitative factors, and generate fundamental-based forecasts.

---

## Section 1: Supply/Demand Balance Construction (30 points)

### Question 1 (15 points)

You're building a crude oil supply/demand balance model. An EIA report states:

> "U.S. crude oil production averaged 13.2 million barrels per day (bpd) last week, up 100,000 bpd from the previous week. Refineries processed 16.5 million bpd of crude, operating at 94% capacity. Crude imports averaged 6.8 million bpd while exports were 4.2 million bpd. Commercial inventories decreased by 5.2 million barrels."

Which balance equation correctly represents this data?

A) Supply = Production + Imports; Demand = Refinery Runs + Exports
B) Inventory Change = (Production + Imports) - (Refinery Runs + Exports)
C) Inventory Change = (Production + Imports - Exports) - Refinery Runs
D) Both B and C are mathematically equivalent and correct

**Answer: D**

**Explanation:**

```python
# Given data:
production = 13.2  # million bpd
imports = 6.8
refinery_runs = 16.5
exports = 4.2
inventory_change = -5.2  # million barrels (decrease)

# Option B: Supply-Demand Balance
# Inventory Change = Total Supply - Total Demand
# Supply = Production + Imports
# Demand = Refinery Runs + Exports

supply = production + imports  # 13.2 + 6.8 = 20.0 million bpd
demand = refinery_runs + exports  # 16.5 + 4.2 = 20.7 million bpd

inventory_change_calc_b = supply - demand  # 20.0 - 20.7 = -0.7 million bpd
# Convert to weekly: -0.7 * 7 = -4.9 million barrels
# Close to reported -5.2 million barrels (difference due to rounding/timing)

# Option C: Net Trade Formulation
# Inventory Change = (Production + Net Imports) - Refinery Runs
# Net Imports = Imports - Exports

net_imports = imports - exports  # 6.8 - 4.2 = 2.6 million bpd
inventory_change_calc_c = production + net_imports - refinery_runs
# = 13.2 + 2.6 - 16.5 = -0.7 million bpd
# = -4.9 million barrels per week ✓

# Both formulations are mathematically equivalent:
# B: (Prod + Imp) - (Ref + Exp)
# C: (Prod + Imp - Exp) - Ref
# Expanding: Prod + Imp - Exp - Ref (same!)

# Why both are useful:
# - Option B: Separates supply sources from demand sinks
# - Option C: Highlights net trade position
```

**Production Implementation:**
```python
@dataclass
class CrudeBalanceSheet:
    """Weekly crude oil balance sheet."""

    # Supply components
    production: float  # million bpd
    imports: float
    stock_draws: float  # Negative of inventory change

    # Demand components
    refinery_runs: float
    exports: float
    stock_builds: float  # Positive inventory change

    # Calculated fields
    @property
    def total_supply(self) -> float:
        return self.production + self.imports + self.stock_draws

    @property
    def total_demand(self) -> float:
        return self.refinery_runs + self.exports + self.stock_builds

    @property
    def balance(self) -> float:
        """Should be ~0 (supply = demand)."""
        return self.total_supply - self.total_demand

    @property
    def net_imports(self) -> float:
        return self.imports - self.exports

    def validate(self):
        """Ensure balance sheet balances."""
        if abs(self.balance) > 0.1:  # tolerance
            raise ValidationError(f"Balance sheet doesn't balance: {self.balance:.2f}")
```

---

### Question 2 (15 points)

You're extracting supply/demand data from a USDA WASDE report using an LLM. The report contains:

```
U.S. Corn Supply and Use (Million Bushels)
                                2023/24    2024/25
Beginning Stocks                 1,377      2,162
Production                      15,342     14,980
Imports                             25         25
    Total Supply                16,744     17,167

Feed and Residual Use            5,425      5,400
Food, Seed & Industrial          6,505      6,555
Exports                          2,150      2,000
    Total Use                   14,080     13,955

Ending Stocks                    2,664      3,212
Stocks-to-Use Ratio              18.9%      23.0%
```

After LLM extraction, you receive:

```json
{
  "2023/24": {
    "beginning_stocks": 1377,
    "production": 15342,
    "imports": 25,
    "total_supply": 16744,
    "total_use": 14080,
    "ending_stocks": 2664,
    "stocks_to_use": 0.189
  },
  "2024/25": {
    "beginning_stocks": 2162,
    "production": 14980,
    "imports": 25,
    "total_supply": 17167,
    "total_use": 13955,
    "ending_stocks": 3212,
    "stocks_to_use": 0.230
  }
}
```

What is the MOST important validation check to run?

A) Verify all numbers are positive
B) Check that 2024/25 beginning stocks equals 2023/24 ending stocks
C) Verify: Ending Stocks = Total Supply - Total Use AND Stocks-to-Use = Ending Stocks / Total Use
D) Confirm production decreased year-over-year (14,980 < 15,342)

**Answer: C**

**Explanation:**

**A - All Numbers Positive:**
- Basic check but insufficient
- Doesn't verify internal consistency
- Statistical discrepancies can be negative (adjustments)

**B - Beginning = Prior Ending:**
```python
# Good check, but there's a problem:
assert data['2024/25']['beginning_stocks'] == data['2023/24']['ending_stocks']
# 2162 == 2664? NO!

# This reveals either:
# 1. LLM extraction error
# 2. USDA revised 2023/24 ending stocks after publication
# 3. Data from different report versions

# Important but not MOST critical for single-period validation
```

**C (Correct) - Accounting Identities:**
```python
def validate_usda_balance_sheet(data):
    """Comprehensive validation of USDA balance sheet."""

    for year, values in data.items():
        # Identity 1: Supply Balance
        # Total Supply = Beginning Stocks + Production + Imports
        calculated_supply = (
            values['beginning_stocks'] +
            values['production'] +
            values['imports']
        )

        if abs(calculated_supply - values['total_supply']) > 1:
            raise ValidationError(
                f"{year}: Total supply doesn't balance. "
                f"Calculated: {calculated_supply}, Reported: {values['total_supply']}"
            )

        # Identity 2: Balance Equation
        # Ending Stocks = Total Supply - Total Use
        calculated_ending = values['total_supply'] - values['total_use']

        if abs(calculated_ending - values['ending_stocks']) > 1:
            raise ValidationError(
                f"{year}: Ending stocks don't balance. "
                f"Calculated: {calculated_ending}, Reported: {values['ending_stocks']}"
            )
            # In our example:
            # 2023/24: 16,744 - 14,080 = 2,664 ✓ (matches)
            # 2024/25: 17,167 - 13,955 = 3,212 ✓ (matches)

        # Identity 3: Stocks-to-Use Ratio
        # S/U = Ending Stocks / Total Use
        calculated_su = values['ending_stocks'] / values['total_use']

        if abs(calculated_su - values['stocks_to_use']) > 0.001:
            raise ValidationError(
                f"{year}: Stocks-to-use ratio incorrect. "
                f"Calculated: {calculated_su:.3f}, Reported: {values['stocks_to_use']:.3f}"
            )
            # 2023/24: 2,664 / 14,080 = 0.189 ✓
            # 2024/25: 3,212 / 13,955 = 0.230 ✓

    # Cross-year validation
    # Beginning stocks should match prior year's ending (with caveats)
    years = sorted(data.keys())
    for i in range(1, len(years)):
        current_beginning = data[years[i]]['beginning_stocks']
        prior_ending = data[years[i-1]]['ending_stocks']

        if abs(current_beginning - prior_ending) > 10:
            # Larger tolerance due to revisions
            warnings.warn(
                f"Beginning stocks mismatch: {years[i]} beginning ({current_beginning}) "
                f"doesn't match {years[i-1]} ending ({prior_ending}). "
                f"Possible revision or data from different report dates."
            )

    return True

# Why this is most critical:
# - Catches LLM arithmetic errors (extracted wrong number)
# - Catches misaligned data (wrong row/column)
# - Ensures balance sheet integrity
# - If these don't balance, the data is unusable
```

**D - YoY Production Change:**
- Directional check
- Doesn't validate data integrity
- Could be right or wrong depending on weather, acreage, yields

**Key Principle:** Always validate accounting identities in commodity balance sheets. If they don't balance, the extraction failed.

---

## Section 2: Qualitative Factor Integration (25 points)

### Question 3 (13 points)

You're building a natural gas winter demand forecast. You have:
- Quantitative: Historical consumption patterns, current storage levels
- Qualitative: LLM-extracted weather outlook from NOAA reports stating "Above-normal temperatures expected across major demand centers"

Which integration approach is MOST rigorous?

**Option A: Qualitative Override**
```python
if llm_weather_outlook == "above_normal_temps":
    demand_forecast *= 0.9  # Reduce by 10%
```

**Option B: Quantitative with Qualitative Context**
```python
# Quantitative model
base_demand = quantitative_forecast(storage, historical_patterns)

# Qualitative context added to output
analysis = f"""Base demand forecast: {base_demand} Bcf/d
Context: NOAA expects above-normal temperatures, which may reduce heating demand.
"""
```

**Option C: Structured Qualitative Factors**
```python
# Extract structured weather outlook
weather_factors = llm_extract_weather(noaa_report)
# {
#   "temp_outlook": "above_normal",
#   "probability": 0.65,
#   "affected_regions": ["Northeast", "Midwest"],
#   "magnitude": "1-2°F above normal"
# }

# Quantify impact using historical relationships
temp_sensitivity = calculate_sensitivity(historical_data)
# Historical: 1°F above normal → -2% demand

temp_deviation = 1.5  # degrees F (from "1-2°F above normal")
demand_adjustment = temp_deviation * temp_sensitivity * 0.65  # probability-weighted
# = 1.5 * -0.02 * 0.65 = -0.0195 (-1.95% demand adjustment)

final_demand = base_demand * (1 + demand_adjustment)

# Confidence adjustment
confidence = base_confidence * weather_factors["probability"]
```

**Option D: LLM Generates Forecast Directly**
```python
forecast = llm.generate(f"Given storage of {storage} Bcf and above-normal temps, what is winter demand?")
```

A) Option A
B) Option B
C) Option C
D) Option D

**Answer: C**

**Explanation:**

**Option A - Qualitative Override:**
```python
# Problems:
# 1. Arbitrary 10% adjustment (no historical basis)
# 2. Treats "above-normal" as binary (ignores magnitude)
# 3. Ignores probability (65% chance vs 90% chance treated same)
# 4. No regional specificity (nationwide vs. Northeast only)
# 5. Not auditable or backtestable
```

**Option B - Context Only:**
```python
# Better than A, but:
# 1. Doesn't quantify impact
# 2. Leaves interpretation to human
# 3. Can't be backtested systematically
# 4. Not actionable for automated trading
```

**Option C (Correct) - Structured Integration:**
```python
def integrate_qualitative_weather(base_forecast, noaa_report, historical_data):
    """Rigorously integrate weather outlook into demand forecast."""

    # Step 1: Extract structured qualitative factors
    weather_prompt = """Extract weather outlook from this NOAA report:

Report: {report}

Return JSON:
{
  "temp_outlook": "above_normal|normal|below_normal",
  "temp_magnitude_f": <expected deviation in °F>,
  "probability": <0-1>,
  "affected_regions": ["region names"],
  "timeframe": "<when this applies>",
  "confidence": <0-1>
}
"""

    weather_factors = llm_extract(weather_prompt, noaa_report)
    # {
    #   "temp_outlook": "above_normal",
    #   "temp_magnitude_f": 1.5,
    #   "probability": 0.65,
    #   "affected_regions": ["Northeast", "Midwest"],
    #   "timeframe": "December-February",
    #   "confidence": 0.75
    # }

    # Step 2: Quantify based on historical relationships
    # Analyze: When temps were X°F above normal, how did demand change?

    sensitivity_by_region = calculate_temp_sensitivity(historical_data)
    # {
    #   "Northeast": -0.025,  # -2.5% demand per °F
    #   "Midwest": -0.020,
    #   "South": -0.010,
    #   "West": -0.015
    # }

    # Step 3: Calculate weighted impact
    total_demand_adjustment = 0
    for region in weather_factors["affected_regions"]:
        region_sensitivity = sensitivity_by_region[region]
        region_demand_share = get_regional_demand_share(region)  # e.g., 0.30 for Northeast

        regional_impact = (
            weather_factors["temp_magnitude_f"] *
            region_sensitivity *
            region_demand_share
        )
        total_demand_adjustment += regional_impact

    # Weight by probability
    expected_adjustment = total_demand_adjustment * weather_factors["probability"]

    # Step 4: Apply to base forecast
    adjusted_forecast = base_forecast * (1 + expected_adjustment)

    # Step 5: Adjust confidence
    # Lower confidence when qualitative factors are uncertain
    forecast_confidence = (
        base_model_confidence * 0.6 +  # Base model weight
        weather_factors["confidence"] * weather_factors["probability"] * 0.4  # Weather factor weight
    )

    return {
        "forecast": adjusted_forecast,
        "base_forecast": base_forecast,
        "weather_adjustment": expected_adjustment,
        "confidence": forecast_confidence,
        "explanation": f"Base forecast adjusted by {expected_adjustment:.1%} due to "
                      f"{weather_factors['temp_magnitude_f']:.1f}°F {weather_factors['temp_outlook']} "
                      f"temperatures (probability: {weather_factors['probability']:.0%})"
    }

# Benefits:
# 1. Quantitative: Historical relationships, not arbitrary adjustments
# 2. Probabilistic: Accounts for forecast uncertainty
# 3. Regional: Weights by affected areas
# 4. Auditable: Every step is documented and verifiable
# 5. Backtestable: Can evaluate historical forecast accuracy
# 6. Confidence-aware: Reduces confidence when uncertainty is high
```

**Option D - LLM Direct Forecast:**
```python
# Problems:
# 1. Hallucination risk (may invent plausible but wrong numbers)
# 2. No systematic methodology
# 3. Can't be backtested or validated
# 4. Black box (can't explain reasoning)
# 5. Unreliable for trading decisions
```

**Key Principle:** Integrate qualitative factors by:
1. Structuring them with LLMs
2. Quantifying impact using historical relationships
3. Weighting by probability/confidence
4. Maintaining auditability and explainability

---

### Question 4 (12 points)

An LLM extracts this qualitative insight from an oil industry report:

> "Major offshore projects in Brazil and Guyana are expected to add 1-1.5 million bpd of production capacity by mid-2025, though some analysts question whether operators can achieve nameplate capacity given technical challenges."

How should this be integrated into a supply forecast model?

A) Add 1.25 million bpd (midpoint of range) to 2025 production forecast
B) Extract structured data, apply historical ramp-up curves, probability-weight scenarios
C) Ignore qualitative forward-looking statements; only use actual production data
D) Ask LLM to generate a production forecast incorporating this information

**Answer: B**

**Explanation:**

**A - Use Midpoint:**
```python
# Problems:
# 1. Ignores "technical challenges" uncertainty
# 2. Assumes immediate full capacity (no ramp-up time)
# 3. Doesn't account for "some analysts question" skepticism
# 4. No probabilistic treatment of range (1-1.5)
```

**B (Correct) - Structured Scenario Analysis:**
```python
def integrate_new_project_capacity(report_text, current_forecast, historical_data):
    """Integrate qualitative capacity outlook into forecast."""

    # Step 1: Extract structured project data
    project_prompt = """Extract new production project information:

Text: {text}

Return JSON array:
[
  {
    "project_name": "<name if mentioned>",
    "location": "<country/region>",
    "capacity_min": <minimum capacity, million bpd>,
    "capacity_max": <maximum capacity, million bpd>,
    "timing": "<when capacity comes online>",
    "uncertainty_factors": ["<factors that could impact delivery>"],
    "analyst_skepticism": <0-1, level of skepticism expressed>
  }
]
"""

    projects = llm_extract(project_prompt, report_text)
    # [
    #   {
    #     "project_name": "Brazil and Guyana offshore",
    #     "location": "South America offshore",
    #     "capacity_min": 1.0,
    #     "capacity_max": 1.5,
    #     "timing": "mid-2025",
    #     "uncertainty_factors": ["technical challenges", "nameplate capacity questions"],
    #     "analyst_skepticism": 0.4
    #   }
    # ]

    # Step 2: Apply historical ramp-up patterns
    # New offshore projects typically achieve:
    # - Month 1-3: 40% of capacity
    # - Month 4-6: 65% of capacity
    # - Month 7-12: 85% of capacity
    # - Year 2+: 95% of capacity (rarely 100%)

    ramp_up_curve = get_historical_ramp_up_curve(
        project_type="offshore",
        historical_projects=historical_data
    )
    # {
    #   "month_3": 0.40,
    #   "month_6": 0.65,
    #   "month_12": 0.85,
    #   "month_24": 0.95
    # }

    # Step 3: Scenario analysis
    for project in projects:
        # Scenario 1: Optimistic (high capacity, smooth ramp)
        optimistic_capacity = project["capacity_max"] * ramp_up_curve["month_12"]
        optimistic_prob = 0.20  # 20% chance of best case

        # Scenario 2: Base case (mid capacity, typical ramp)
        base_capacity_nameplate = (project["capacity_min"] + project["capacity_max"]) / 2
        base_capacity = base_capacity_nameplate * ramp_up_curve["month_12"]
        base_prob = 0.50  # 50% chance

        # Scenario 3: Pessimistic (low capacity, technical issues)
        # Account for analyst skepticism
        pessimistic_capacity = project["capacity_min"] * ramp_up_curve["month_6"]  # Slower ramp
        pessimistic_capacity *= (1 - project["analyst_skepticism"] * 0.5)  # Further reduction
        pessimistic_prob = 0.30  # 30% chance

        # Expected value
        expected_capacity = (
            optimistic_capacity * optimistic_prob +
            base_capacity * base_prob +
            pessimistic_capacity * pessimistic_prob
        )

        # Example calculation:
        # Optimistic: 1.5 * 0.85 * 0.20 = 0.255
        # Base: 1.25 * 0.85 * 0.50 = 0.531
        # Pessimistic: 1.0 * 0.65 * 0.80 * 0.30 = 0.156
        # Expected: 0.942 million bpd

        # Step 4: Integrate into forecast with timing
        timing_date = parse_timing(project["timing"])  # "mid-2025" → 2025-07-01

        for forecast_month in current_forecast:
            if forecast_month["date"] >= timing_date:
                months_since_start = (forecast_month["date"] - timing_date).days / 30

                # Apply ramp-up curve
                if months_since_start <= 3:
                    ramp_factor = ramp_up_curve["month_3"]
                elif months_since_start <= 6:
                    ramp_factor = ramp_up_curve["month_6"]
                elif months_since_start <= 12:
                    ramp_factor = ramp_up_curve["month_12"]
                else:
                    ramp_factor = ramp_up_curve["month_24"]

                # Adjust expected capacity by ramp-up progress
                incremental_production = expected_capacity * (ramp_factor / ramp_up_curve["month_12"])

                forecast_month["production"] += incremental_production
                forecast_month["production_breakdown"]["new_projects"] += incremental_production

                # Track uncertainty
                forecast_month["uncertainty_range_mbpd"] += (
                    optimistic_capacity - pessimistic_capacity
                )

    return current_forecast

# Result for mid-2025:
# - Month 1 (Jul 2025): +0.38 million bpd (40% of 0.942)
# - Month 6 (Dec 2025): +0.61 million bpd (65% of 0.942)
# - Month 12 (Jun 2026): +0.80 million bpd (85% of 0.942)
# - Uncertainty range: ±0.4 million bpd

# vs. Naive approach (A): +1.25 million bpd immediately
```

**C - Ignore Forward-Looking:**
```python
# Too conservative
# - Misses valuable information
# - Forecasts will lag actual developments
# - Markets price in expectations, not just current reality
```

**D - LLM Direct Forecast:**
```python
# Same problems as before:
# - Hallucination risk
# - No systematic methodology
# - Can't validate or backtest
```

**Key Principle:** Forward-looking qualitative information is valuable but must be:
1. Structured into scenarios
2. Probability-weighted
3. Adjusted by historical realization rates
4. Timed appropriately (ramp-up curves)

---

## Section 3: Integrated Fundamental Models (25 points)

### Question 5 (15 points)

You're building an integrated fundamental model for crude oil that combines:
- EIA weekly inventory data (quantitative)
- OPEC+ production decisions (qualitative, extracted via LLM)
- Refinery maintenance schedules (qualitative)
- Geopolitical risk assessments (qualitative)

Which model architecture is MOST appropriate?

**Architecture A: Ensemble Approach**
```python
# Separate models for each data type, then average predictions
inventory_model = TimeSeriesModel(eia_data)
opec_model = LLMQualitativeModel(opec_statements)
refinery_model = MaintenanceModel(schedules)
geo_model = RiskModel(geopolitical_data)

final_forecast = average([
    inventory_model.predict(),
    opec_model.predict(),
    refinery_model.predict(),
    geo_model.predict()
])
```

**Architecture B: Feature Engineering + ML**
```python
# Extract structured features from all sources, train ML model
features = {
    "inventory_level": eia_data["inventory"],
    "inventory_vs_avg": eia_data["vs_5yr_avg"],
    "opec_sentiment": llm_extract_sentiment(opec_statements),
    "refinery_offline_capacity": calculate_offline(schedules),
    "geopolitical_risk_score": llm_score_risk(geo_events)
}

model = GradientBoostingRegressor()
model.fit(features, target=price_changes)
forecast = model.predict(current_features)
```

**Architecture C: Hierarchical Causal Model**
```python
# Model causal relationships between fundamentals and price

# Level 1: Fundamental factors
supply = {
    "opec_production": extract_opec_policy(),
    "us_production": eia_data["production"],
    "maintenance_impact": -calculate_offline_capacity()
}
total_supply = sum(supply.values())

demand = {
    "refinery_runs": eia_data["runs"],
    "exports": eia_data["exports"],
    "geopolitical_disruption": estimate_demand_loss_from_conflicts()
}
total_demand = sum(demand.values())

# Level 2: Market balance
fundamental_balance = total_supply - total_demand

# Level 3: Inventory dynamics
inventory_change = fundamental_balance
days_of_cover = inventory_level / (total_demand / 365)

# Level 4: Price impact
if days_of_cover < 25:  # Tight market
    price_pressure = "bullish"
    expected_price_change = calculate_tightness_premium(days_of_cover)
elif days_of_cover > 40:  # Oversupplied
    price_pressure = "bearish"
    expected_price_change = calculate_excess_discount(days_of_cover)
else:
    price_pressure = "neutral"
    expected_price_change = 0

# Adjust for geopolitical risk premium
risk_premium = estimate_risk_premium(geo_events)
final_forecast = current_price * (1 + expected_price_change + risk_premium)
```

**Architecture D: LLM-Generated Analysis**
```python
context = f"""
EIA Data: {eia_summary}
OPEC Statements: {opec_text}
Refinery Maintenance: {maintenance_summary}
Geopolitical Events: {geo_summary}
"""

forecast = llm.generate(f"Given this context, forecast crude oil prices: {context}")
```

A) Architecture A
B) Architecture B
C) Architecture C
D) Architecture D

**Answer: C**

**Explanation:**

**Architecture A - Ensemble:**
```python
# Problems:
# 1. Models don't interact (OPEC cuts + refinery maintenance = additive bullish, but averaged)
# 2. Equal weighting inappropriate (inventory more important than geopolitics usually)
# 3. Ignores causal relationships (supply cuts → lower inventory → higher prices)
# 4. Can't explain forecast logic
```

**Architecture B - ML Feature Engineering:**
```python
# Pros:
# - Systematic feature extraction
# - Learns patterns from data
# - Handles non-linear relationships

# Cons:
# - Black box (can't explain "why")
# - Requires large training dataset
# - Doesn't respect fundamental relationships (supply-demand balance)
# - May learn spurious correlations
# - Fails in regime changes (model trained on surplus, fails in shortage)
```

**Architecture C (Correct) - Hierarchical Causal:**
```python
class IntegratedFundamentalModel:
    """Hierarchical causal model for crude oil fundamentals."""

    def __init__(self):
        self.llm = AnthropicLLM()
        self.historical_data = load_historical_data()

    def forecast_price(self, date):
        """Generate fundamental-based price forecast."""

        # ========================================
        # LEVEL 1: Extract Fundamental Factors
        # ========================================

        # Supply factors
        supply = self.calculate_supply(date)
        # {
        #   "opec_production": 28.5,  # million bpd
        #   "non_opec_production": 55.2,
        #   "us_production": 13.2,
        #   "refinery_maintenance_impact": -0.8,  # Reduces effective supply
        #   "total": 96.1
        # }

        # Demand factors
        demand = self.calculate_demand(date)
        # {
        #   "global_consumption": 101.5,  # million bpd
        #   "geopolitical_disruption": -0.5,  # Libya outages
        #   "total": 101.0
        # }

        # ========================================
        # LEVEL 2: Market Balance
        # ========================================

        balance = supply["total"] - demand["total"]
        # -4.9 million bpd (deficit)

        # ========================================
        # LEVEL 3: Inventory Dynamics
        # ========================================

        # Convert balance to inventory change
        implied_inventory_change = balance * 7  # weekly = bpd * 7 days
        # -34.3 million barrels per week (drawdown)

        # Update inventory projection
        current_inventory = self.get_latest_eia_inventory()
        projected_inventory = current_inventory + implied_inventory_change

        # Calculate days of cover
        days_of_cover = (projected_inventory / (demand["total"] * 1_000_000)) * 365
        # Assuming 450 million barrel inventory: 450 / 101 = 4.5 months = ~32 days

        # ========================================
        # LEVEL 4: Price Impact Model
        # ========================================

        # Historical relationship: days of cover vs price premium
        # - 45+ days: -10% to -5% (well-supplied)
        # - 35-45 days: -5% to 0% (adequately supplied)
        # - 25-35 days: 0% to +5% (balanced)
        # - 15-25 days: +5% to +15% (tight)
        # - <15 days: +15% to +30% (crisis)

        if days_of_cover < 15:
            base_premium = 0.20  # +20%
        elif days_of_cover < 25:
            base_premium = 0.10 + (25 - days_of_cover) / 10 * 0.10
        elif days_of_cover < 35:
            base_premium = (35 - days_of_cover) / 10 * 0.10
        elif days_of_cover < 45:
            base_premium = -(days_of_cover - 35) / 10 * 0.05
        else:
            base_premium = -0.05 - (days_of_cover - 45) / 10 * 0.05

        # ========================================
        # LEVEL 5: Qualitative Adjustments
        # ========================================

        # Geopolitical risk premium
        geo_events = self.fetch_geopolitical_events(date)
        geo_premium = self.assess_risk_premium(geo_events)
        # Uses LLM to extract severity and probability

        # OPEC policy uncertainty
        opec_statements = self.fetch_opec_communications(date)
        policy_uncertainty = self.assess_policy_uncertainty(opec_statements)

        # Combined adjustment
        total_premium = base_premium + geo_premium
        confidence = 1.0 - policy_uncertainty  # Higher uncertainty → lower confidence

        # ========================================
        # LEVEL 6: Price Forecast
        # ========================================

        current_price = self.get_current_price()
        forecast_price = current_price * (1 + total_premium)

        return {
            "forecast_price": forecast_price,
            "current_price": current_price,
            "implied_change": total_premium,
            "confidence": confidence,
            "drivers": {
                "supply_demand_balance": balance,
                "inventory_days_cover": days_of_cover,
                "fundamental_premium": base_premium,
                "geopolitical_premium": geo_premium,
            },
            "explanation": self.generate_explanation(supply, demand, balance, days_of_cover, geo_premium)
        }

    def generate_explanation(self, supply, demand, balance, days_cover, geo_premium):
        """Human-readable explanation of forecast."""

        explanation = f"""
Fundamental Analysis:

SUPPLY: {supply['total']:.1f} million bpd
- OPEC: {supply['opec_production']:.1f} million bpd
- Non-OPEC: {supply['non_opec_production']:.1f} million bpd
- US: {supply['us_production']:.1f} million bpd
- Refinery maintenance impact: {supply['refinery_maintenance_impact']:.1f} million bpd

DEMAND: {demand['total']:.1f} million bpd
- Global consumption: {demand['global_consumption']:.1f} million bpd
- Disruptions: {demand['geopolitical_disruption']:.1f} million bpd

BALANCE: {balance:+.1f} million bpd ({"Deficit" if balance < 0 else "Surplus"})

INVENTORY OUTLOOK:
- Days of cover: {days_cover:.0f} days
- Market condition: {"Tight" if days_cover < 25 else "Balanced" if days_cover < 35 else "Well-supplied"}
- Fundamental price impact: {base_premium:+.1%}

RISK FACTORS:
- Geopolitical premium: {geo_premium:+.1%}

CONCLUSION:
{"Bullish" if total_premium > 0.05 else "Bearish" if total_premium < -0.05 else "Neutral"} outlook based on fundamental balance and inventory dynamics.
"""
        return explanation

# Benefits of this architecture:
# 1. Causal: Respects supply → inventory → price relationships
# 2. Explainable: Every step is transparent and auditable
# 3. Flexible: Can weight factors based on market regime
# 4. Robust: Doesn't rely on ML training data
# 5. Integrates LLMs: Uses LLMs for qualitative extraction, not forecasting
# 6. Backtest-friendly: Can validate historical relationships
```

**Architecture D - LLM Direct:**
- Same problems: hallucination risk, no systematic approach

**Key Principle:** Build causal models that respect fundamental relationships. Use LLMs to extract and structure inputs, not generate forecasts directly.

---

### Question 6 (10 points)

Your fundamental model forecasts crude oil prices will rise 10% over the next month based on tight supply-demand balance. However, crude oil futures are already in steep backwardation (prompt month $85, 6-month $75). What does this suggest?

A) Your model is wrong; the market disagrees
B) The market has already priced in the tightness; your forecast may be late
C) Arbitrage opportunity - buy futures and profit from your forecast
D) Backwardation confirms your bullish view; increase position size

**Answer: B**

**Explanation:**

**Understanding Backwardation:**
```python
# Market structure:
# - Spot/prompt: $85/bbl
# - 6-month futures: $75/bbl
# - Backwardation: Near > Far (inverted curve)

# Backwardation signals:
# 1. Tight current supply (high premium for immediate delivery)
# 2. Expectation of loosening (lower future prices)
# 3. Strong carry incentive (release storage now, not later)
```

**A - Model is Wrong:**
```python
# Not necessarily
# - Model captures current fundamentals correctly (tight market)
# - But market is forward-looking, model may be backward-looking
```

**B (Correct) - Already Priced In:**
```python
# Analysis:
# Your model: "Tight supply → prices should rise 10%"
# Market: "Yes, it's tight NOW ($85), but will ease later ($75)"

# The $85 prompt price ALREADY reflects the tightness
# Your model identified the tightness correctly
# But you're late - the market priced this in before you did

# Key questions:
# 1. WHEN did the market price in tightness?
# 2. WHY does the market expect loosening (6-month $75)?

def analyze_market_expectations(futures_curve, fundamental_model):
    """Compare model forecast to market expectations."""

    # Current market prices
    spot = futures_curve["prompt"]  # $85
    forward_6m = futures_curve["6_month"]  # $75

    # Market-implied expectation
    market_implied_change = (forward_6m - spot) / spot
    # ($75 - $85) / $85 = -11.8% (market expects prices to fall)

    # Your model forecast
    model_forecast = fundamental_model.forecast(horizon="6_month")
    model_implied_change = +0.10  # +10%

    # Divergence
    divergence = model_implied_change - market_implied_change
    # +10% - (-11.8%) = +21.8% (huge difference!)

    # Possible explanations:
    if divergence > 0.15:  # Model more bullish than market
        explanations = [
            # 1. Market sees future supply increases you don't
            "Check: Are new production projects coming online that your model misses?",

            # 2. Market expects demand destruction
            "Check: Does market expect recession/demand slowdown?",

            # 3. Market sees geopolitical resolution
            "Check: Are tensions expected to ease?",

            # 4. Your model is backward-looking
            "Check: Is your model based on current data while market prices future?",

            # 5. Positioning/technical factors
            "Check: Is backwardation due to short-term squeeze, not true tightness?"
        ]

    return {
        "market_expectation": market_implied_change,
        "model_expectation": model_implied_change,
        "divergence": divergence,
        "possible_explanations": explanations
    }

# In this case:
# - Your model correctly identifies CURRENT tightness
# - But market expects future loosening (OPEC+ reversing cuts? Demand slowdown?)
# - Your forecast is "stale" - based on current fundamentals, not future changes

# Action:
# 1. Investigate WHY market expects $75 in 6 months
# 2. Update model with forward-looking factors market is pricing
# 3. Don't blindly trade against market without understanding divergence
```

**C - Arbitrage Opportunity:**
```python
# DANGEROUS assumption
# - Markets are generally efficient
# - If you see "obvious" arbitrage, ask why it exists
# - Likely you're missing information, not smarter than the market

# Real arbitrage conditions:
# 1. Risk-free profit (backwardation itself is not risk-free)
# 2. No capital constraints
# 3. No storage/carry costs

# In commodities:
# - Backwardation reflects strong spot demand
# - If you buy futures at $75, you still have risk prices stay high
# - Not true arbitrage
```

**D - Confirm Bullish View:**
```python
# Wrong interpretation
# - Backwardation confirms CURRENT tightness (which your model also sees)
# - But forward curve at $75 suggests market expects loosening
# - If anything, this should make you cautious, not aggressive

# Increasing position size without understanding why market disagrees is risky
```

**Key Lesson:**
```python
# When your model diverges from market prices:
# 1. Understand WHY (not just THAT) you disagree
# 2. Market may know something you don't (future supply, demand changes)
# 3. Your edge must be in information/analysis market lacks
# 4. Steep backwardation often means market has ALREADY priced tightness

# Better approach:
# - If you believe market is wrong about future loosening, trade the spread:
#   - Sell prompt, buy 6-month (bet on curve flattening)
# - Lower risk than outright long position
# - Profit if your view (tightness persists) is correct
```

---

## Section 4: Forecast Validation (20 points)

### Question 7 (10 points)

You've generated fundamental forecasts for crude oil inventory changes over 12 weeks. Results:

```
Forecast vs Actual Inventory Change (million barrels)
Week  Forecast  Actual  Error
1     -4.5      -5.2    +0.7
2     -3.2      -2.8    -0.4
3     -2.1      -1.5    -0.6
4     +0.5      -1.2    +1.7
5     +2.3      +1.8    +0.5
6     +1.5      +2.1    -0.6
7     -0.8      +0.3    -1.1
8     -2.5      -3.1    +0.6
9     -1.8      -0.9    -0.9
10    +0.2      +1.5    -1.3
11    +3.1      +2.8    +0.3
12    +2.5      +3.0    -0.5

RMSE: 0.86 million barrels
Mean Error: 0.03 million barrels (nearly unbiased)
Hit Rate (correct direction): 75%
```

Which assessment is MOST accurate?

A) Excellent model - 75% hit rate and low bias
B) Good accuracy (RMSE < 1mb) but investigate weeks 4, 7, 10 (large errors)
C) Model has systematic bias - consistently misses turning points
D) Unreliable - cannot use for trading

**Answer: B**

**Explanation:**

**Model Performance Analysis:**
```python
def analyze_forecast_performance(forecasts, actuals):
    """Comprehensive forecast evaluation."""

    errors = forecasts - actuals

    # Metric 1: RMSE (magnitude of errors)
    rmse = np.sqrt(np.mean(errors**2))
    # 0.86 million barrels - reasonable for weekly inventory forecasts
    # (Typical changes: -5 to +5 mb, so 0.86 error is ~17% of range)

    # Metric 2: Bias (systematic over/under prediction)
    mean_error = np.mean(errors)
    # 0.03 mb - essentially unbiased ✓

    # Metric 3: Directional accuracy
    correct_direction = np.mean(np.sign(forecasts) == np.sign(actuals))
    # 75% - good (50% is random) ✓

    # Metric 4: Absolute Mean Error
    mae = np.mean(np.abs(errors))

    # Metric 5: Identify systematic failures
    large_errors = errors[np.abs(errors) > 1.0]
    large_error_weeks = np.where(np.abs(errors) > 1.0)[0]
    # Weeks 4, 7, 10

    return {
        "rmse": rmse,
        "mean_error": mean_error,
        "mae": mae,
        "hit_rate": correct_direction,
        "large_error_weeks": large_error_weeks,
        "assessment": classify_performance(rmse, mean_error, correct_direction)
    }

# Detailed investigation of large error weeks:
def investigate_large_errors(week_4, week_7, week_10):
    """What caused large forecast errors?"""

    # Week 4: Forecast +0.5 mb, Actual -1.2 mb (Error +1.7)
    # - Forecast: Small build expected
    # - Actual: Drawdown
    # Possible causes:
    # 1. Refinery demand higher than expected
    # 2. Imports lower than forecast
    # 3. Exports higher than forecast
    # 4. LLM missed key factor in qualitative data

    # Week 7: Forecast -0.8 mb, Actual +0.3 mb (Error -1.1)
    # - Forecast: Drawdown
    # - Actual: Small build
    # Possible causes:
    # 1. Refinery maintenance (reduced demand)
    # 2. Import surge
    # 3. Weather disruption to demand

    # Week 10: Forecast +0.2 mb, Actual +1.5 mb (Error -1.3)
    # - Forecast: Small build
    # - Actual: Large build
    # Possible causes:
    # 1. Demand destruction (recession fears)
    # 2. Production surge
    # 3. Refinery outages

    # Common pattern: Turning points
    # - Week 4: Trend changed from drawdown to build season
    # - Week 7: Mid-season volatility
    # - Week 10: Build season accelerating

    # Model weakness: Slow to adapt to changing conditions
    return "Model struggles with turning points and regime changes"

# Recommendations:
# 1. Add regime detection (is market transitioning?)
# 2. Improve refinery maintenance tracking (major demand factor)
# 3. Add uncertainty estimates (wider confidence bands at turning points)
# 4. Post-mortem large errors to improve qualitative factor extraction
```

**A - Excellent Model:**
```python
# Too generous
# - 75% hit rate is good, not excellent
# - Large errors at key weeks are concerning
# - Model is useful but has clear improvement areas
```

**B (Correct) - Good with Targeted Improvements:**
```python
# Realistic assessment:

# Strengths:
# ✓ Unbiased (mean error ~0)
# ✓ Reasonable RMSE (0.86 mb)
# ✓ Good directional accuracy (75%)

# Weaknesses:
# ✗ Large errors at turning points (weeks 4, 7, 10)
# ✗ May struggle with regime changes

# Actionable:
# 1. Investigate why forecasts failed those specific weeks
# 2. Identify patterns (all three were near seasonal transitions?)
# 3. Improve model for those conditions
# 4. Add confidence bands (wider during uncertain periods)

# Trading implications:
# - Model is reliable for trend-following
# - Reduce position size during likely turning points
# - Use model as input, not absolute truth
```

**C - Systematic Bias at Turning Points:**
```python
# Partially correct observation but overstated
# - Yes, large errors cluster around turning points
# - But 75% hit rate means model is right more often than wrong
# - "Systematic" implies consistent failure pattern
# - Here, only 3 out of 12 weeks had large errors (25%)
```

**D - Unreliable:**
```python
# Too harsh
# - 75% directional accuracy is tradeable
# - RMSE of 0.86 mb is reasonable for weekly forecasts
# - Many profitable trading strategies have <70% win rates
# - Model is useful with proper risk management
```

**Key Principle:** Evaluate models on multiple metrics. Perfect forecasts are impossible; focus on identifying and improving specific failure modes.

---

### Question 8 (10 points)

Your fundamental model is used to generate weekly trading signals. After 6 months:
- Model forecasts: 60% directional accuracy
- Trading P&L: Consistently positive but with occasional large losses
- Largest loss: Week when your model forecast drawdown, but unexpected OPEC+ announcement caused price spike

How should you improve the system?

A) Improve model accuracy to 70%+ before trading again
B) Add position sizing based on forecast confidence + add stop-losses for unexpected events
C) Switch to machine learning model trained on historical announcements
D) Only trade when model confidence is 90%+ (very few trades)

**Answer: B**

**Explanation:**

**A - Improve Accuracy First:**
```python
# Problems:
# 1. 60% directional accuracy is already profitable with good risk management
# 2. Improving to 70% may take months/years of research
# 3. Forgoes profitable trading opportunity
# 4. No model will predict unexpected announcements (OPEC+ surprises)

# Reality: Even 55% accuracy is profitable with proper position sizing
```

**B (Correct) - Risk Management First:**
```python
class FundamentalTradingSystem:
    """Integrate fundamental forecasts with risk management."""

    def generate_trading_signal(self, fundamental_forecast, current_position):
        """Convert forecast to position with risk management."""

        # Step 1: Assess forecast confidence
        # (based on model track record, data quality, etc.)
        forecast_confidence = self.assess_confidence(fundamental_forecast)

        # Step 2: Position sizing based on confidence
        # Kelly criterion variant for commodity trading
        edge = 0.60  # 60% win rate
        win_loss_ratio = 1.5  # Avg win / avg loss

        kelly_fraction = (edge * win_loss_ratio - (1 - edge)) / win_loss_ratio
        # = (0.60 * 1.5 - 0.40) / 1.5 = 0.33

        # Conservative: Use fractional Kelly (25% of Kelly)
        base_position_size = kelly_fraction * 0.25 * self.capital
        # = 0.33 * 0.25 * $1M = $82,500

        # Adjust for forecast confidence
        if forecast_confidence > 0.7:
            position_multiplier = 1.2  # Increase size
        elif forecast_confidence < 0.5:
            position_multiplier = 0.5  # Reduce size
        else:
            position_multiplier = 1.0

        target_position = base_position_size * position_multiplier

        # Step 3: Implement stop-losses for unexpected events
        # Problem: OPEC+ surprise announcement
        # Solution: Don't let single event wipe out multiple weeks of profits

        stop_loss_pct = 0.02  # 2% of position
        max_position_loss = target_position * stop_loss_pct

        # Step 4: Diversify across time
        # Don't put full position on immediately
        # Scale in over 2-3 days to reduce timing risk

        daily_scaling = target_position / 3

        # Step 5: Event risk monitoring
        # Reduce position before known event risk dates

        upcoming_events = self.check_event_calendar()
        # {"OPEC+ meeting": "2024-12-05", "EIA report": "2024-12-01"}

        if any(self.days_until(event) < 2 for event in upcoming_events.values()):
            # Major event within 2 days
            position_multiplier *= 0.5  # Half position into event risk

        return {
            "target_position": target_position,
            "daily_scale_in": daily_scaling,
            "stop_loss": current_price * (1 - 0.02),  # 2% stop
            "rationale": fundamental_forecast["explanation"],
            "confidence": forecast_confidence,
            "risk_adjustments": {
                "confidence_multiplier": position_multiplier,
                "event_risk_reduction": 0.5 if upcoming_events else 1.0
            }
        }

    def handle_stop_loss_hit(self, position, reason):
        """What to do when stop is triggered."""

        # Log for analysis
        self.log_stop_loss(position, reason)

        # Exit position
        self.close_position(position)

        # Review: Was this a model failure or external shock?
        if reason == "unexpected_announcement":
            # External shock (OPEC+ surprise)
            # → Not a model failure
            # → Resume trading after event passes

            # Wait for market to digest announcement
            self.pause_trading(days=2)

        elif reason == "model_forecast_wrong":
            # Model forecast was incorrect
            # → Potential model issue
            # → Review model assumptions

            self.flag_for_model_review()

        return

# Results with this approach:
# - Occasional large losses prevented by 2% stops
# - Smaller position sizes into event risk
# - Steady compounding of edges over time
# - 60% win rate + 1.5:1 win/loss ratio = profitable system
```

**C - ML Model for Announcements:**
```python
# Problems:
# 1. OPEC+ announcements are rare (quarterly at most)
# 2. Each announcement is in different geopolitical context
# 3. Insufficient data to train reliable ML model
# 4. Announcements are often surprises by design

# ML works for frequent, consistent patterns
# Geopolitical announcements are sporadic and unique
```

**D - Only Trade High Confidence:**
```python
# Problems:
# 1. Very few trades → can't capitalize on edge
# 2. High confidence doesn't guarantee correctness
# 3. Opportunity cost of sitting out 90% of the time
# 4. May miss best opportunities (sometimes uncertainty creates opportunity)

# Better approach: Scale position size with confidence, not trade frequency
```

**Key Principle:** Trading systems require risk management as much as forecast accuracy. Protect profits with stops, size positions by confidence, and reduce exposure into known event risk.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | D | 15 | Supply/demand balance equations |
| 2 | C | 15 | USDA balance sheet validation |
| 3 | C | 13 | Structured qualitative integration |
| 4 | B | 12 | New capacity project integration |
| 5 | C | 15 | Hierarchical causal model architecture |
| 6 | B | 10 | Market pricing vs. model forecast |
| 7 | B | 10 | Forecast accuracy evaluation |
| 8 | B | 10 | Trading system risk management |

**Total:** 100 points

---

## Grading Scale

- **90-100:** Excellent - Advanced fundamentals expertise
- **80-89:** Good - Ready for production implementation
- **70-79:** Adequate - Review integration and validation
- **Below 70:** Needs improvement - Revisit balance sheet construction

---

## Key Takeaways

1. **Validate accounting identities** - Balance sheets must balance
2. **Structure qualitative factors** - Quantify via historical relationships
3. **Build causal models** - Respect supply → inventory → price logic
4. **Market prices are forward-looking** - Your model may be late
5. **Risk management > forecast accuracy** - 60% edge + good risk = profitable

---

## Next Steps

**Score 90-100:** Proceed to Module 5 (Signal Generation)
**Score 80-89:** Review qualitative integration, then proceed
**Score 70-79:** Practice building balance sheet models
**Score <70:** Revisit Module 4, focus on fundamentals
