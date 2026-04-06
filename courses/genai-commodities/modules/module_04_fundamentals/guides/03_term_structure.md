# Term Structure Analysis with LLMs

> **Reading time:** ~10 min | **Module:** Module 4: Fundamentals | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Term structure analysis examines the relationship between futures contracts of different maturities, revealing market expectations about future price levels, storage costs, and convenience yield. LLMs enhance this analysis by extracting narrative context from news, reports, and commentary to expl...

</div>

## In Brief

Term structure analysis examines the relationship between futures contracts of different maturities, revealing market expectations about future price levels, storage costs, and convenience yield. LLMs enhance this analysis by extracting narrative context from news, reports, and commentary to explain term structure shapes (contango vs backwardation) and predict structural shifts.

<div class="callout-insight">

**Insight:** The futures curve shape tells a story: contango suggests plentiful supply (willing to pay storage), backwardation signals scarcity (immediate need outweighs future value). LLMs read this story in reverse—analyzing news, inventories, and market commentary to predict whether the curve will steepen, flatten, or invert. This combines quantitative curve modeling with qualitative fundamental understanding.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of airline ticket pricing:

**Contango (normal):**
- Today's flight: $200
- Flight in 3 months: $300
- "Storage cost" = uncertainty, demand risk
- Airlines charge more for future flexibility

**Backwardation (unusual):**
- Today's flight: $500 (urgent trip!)
- Flight in 3 months: $250
- Immediate need drives today's price up
- Future has more capacity/flexibility

For crude oil:

**Contango scenario:**
- Current supply: Comfortable
- Inventories: Above average
- Market: Willing to store for future
- Curve: Front month $70, 12-month $75
- Implication: No immediate shortage

**Backwardation scenario:**
- Current supply: Tight
- Inventories: Below average
- Market: Values immediate barrels highly
- Curve: Front month $80, 12-month $75
- Implication: Shortage now, expected relief later

**LLM adds:**
- "Why is the market in backwardation?"
  → LLM reads: "OPEC announced surprise production cut"
- "Will backwardation persist?"
  → LLM analyzes: Upcoming production increases, inventory builds

## Formal Definition

### Futures Term Structure

**Futures prices at time t for delivery at T:**
$$F(t, T_1), F(t, T_2), ..., F(t, T_n)$$

where $T_1 < T_2 < ... < T_n$

**Spot-futures relationship:**
$$F(t, T) = S_t e^{(r + c - y)(T - t)}$$

Where:
- $S_t$: Spot price
- $r$: Risk-free rate
- $c$: Storage cost
- $y$: Convenience yield

**Term structure shapes:**

**1. Contango:** $F(t, T_2) > F(t, T_1)$ for $T_2 > T_1$
- Market expectation: Prices rising or carrying cost > convenience yield
- Interpretation: Supply comfortable, willing to store

**2. Backwardation:** $F(t, T_2) < F(t, T_1)$ for $T_2 > T_1$
- Market expectation: Immediate demand exceeds supply
- Interpretation: Scarcity, high convenience yield

**3. Calendar spread:** $\text{Spread}(T_1, T_2) = F(t, T_2) - F(t, T_1)$
- Positive → contango
- Negative → backwardation

### Key Metrics

**Slope (front-to-back spread):**
$$\text{Slope} = \frac{F(t, T_n) - F(t, T_1)}{T_n - T_1}$$

**Curvature (butterfly):**
$$\text{Butterfly} = F(t, T_2) - \frac{F(t, T_1) + F(t, T_3)}{2}$$

Positive curvature → curve humped in middle

**Roll yield:**
For long position in near contract:
$$\text{Roll Yield} = \frac{F(t, T_1) - F(t-\Delta t, T_1 + \Delta t)}{F(t-\Delta t, T_1 + \Delta t)}$$

Positive in backwardation (near contract more expensive), negative in contango.

## Code Implementation

### Term Structure Extraction and Analysis


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">termstructureanalyzer.py</span>
</div>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from anthropic import Anthropic

class TermStructureAnalyzer:
    """
    Analyze commodity futures term structure with LLM augmentation.
    """

    def __init__(self, anthropic_api_key):
        self.client = Anthropic(api_key=anthropic_api_key)

    def extract_curve(self, futures_data, date):
        """
        Extract term structure for specific date.

        Args:
            futures_data: DataFrame with columns [date, contract, price]
            date: Analysis date

        Returns:
            curve: DataFrame [maturity, price, days_to_expiry]
        """
        # Filter for specific date
        curve_data = futures_data[futures_data['date'] == date].copy()

        # Calculate days to expiry
        curve_data['days_to_expiry'] = (
            pd.to_datetime(curve_data['maturity']) - pd.to_datetime(date)
        ).dt.days

        # Sort by maturity
        curve_data = curve_data.sort_values('days_to_expiry')

        return curve_data[['maturity', 'price', 'days_to_expiry']]

    def compute_metrics(self, curve):
        """
        Compute term structure metrics.

        Args:
            curve: DataFrame from extract_curve()

        Returns:
            metrics: Dictionary of term structure metrics
        """
        prices = curve['price'].values
        days = curve['days_to_expiry'].values

        # Market structure
        if len(prices) < 2:
            return {}

        front_price = prices[0]
        back_price = prices[-1]
        spread = back_price - front_price

        structure = 'contango' if spread > 0 else 'backwardation'

        # Slope (annualized)
        days_diff = days[-1] - days[0]
        slope = (spread / front_price) * (365 / days_diff) if days_diff > 0 else 0

        # Calendar spreads
        calendar_spreads = {}
        if len(prices) >= 2:
            for i in range(len(prices) - 1):
                spread_name = f"M{i+1}-M{i+2}"
                calendar_spreads[spread_name] = prices[i+1] - prices[i]

        # Curvature (butterfly) if enough contracts
        curvature = None
        if len(prices) >= 3:
            # Front butterfly: M2 - (M1 + M3)/2
            curvature = prices[1] - (prices[0] + prices[2]) / 2

        metrics = {
            'structure': structure,
            'front_price': front_price,
            'back_price': back_price,
            'front_back_spread': spread,
            'slope_annualized': slope,
            'calendar_spreads': calendar_spreads,
            'curvature': curvature
        }

        return metrics

    def visualize_curve(self, curve, metrics=None, title=None):
        """
        Visualize term structure.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Price curve
        ax1.plot(curve['days_to_expiry'], curve['price'],
                'o-', linewidth=2, markersize=8, label='Futures prices')
        ax1.axhline(curve['price'].iloc[0], color='red', linestyle='--',
                   alpha=0.5, label='Spot reference')
        ax1.set_xlabel('Days to Expiry')
        ax1.set_ylabel('Price ($/unit)')
        ax1.set_title(title or 'Futures Term Structure')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Add metrics annotation
        if metrics:
            textstr = f"Structure: {metrics['structure'].upper()}\n"
            textstr += f"Front: ${metrics['front_price']:.2f}\n"
            textstr += f"Back: ${metrics['back_price']:.2f}\n"
            textstr += f"Spread: ${metrics['front_back_spread']:.2f}\n"
            textstr += f"Slope: {metrics['slope_annualized']*100:.2f}%"

            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Calendar spreads
        if metrics and metrics.get('calendar_spreads'):
            spreads = metrics['calendar_spreads']
            spread_names = list(spreads.keys())
            spread_values = list(spreads.values())

            colors = ['green' if v > 0 else 'red' for v in spread_values]

            ax2.bar(spread_names, spread_values, color=colors, alpha=0.7)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Calendar Spread')
            ax2.set_ylabel('Spread Value ($)')
            ax2.set_title('Calendar Spreads (Contango = Positive)')
            ax2.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def llm_interpret_structure(self, curve, metrics, news_context):
        """
        Use LLM to interpret term structure in context.

        Args:
            curve: Term structure data
            metrics: Computed metrics
            news_context: List of recent news articles

        Returns:
            interpretation: LLM analysis of term structure
        """
        # Format curve data
        curve_str = curve.to_string(index=False)

        # Format news
        news_str = "\n\n".join(news_context[:5])

        prompt = f"""Analyze this commodity futures term structure.

TERM STRUCTURE DATA:
{curve_str}

STRUCTURE METRICS:
- Type: {metrics['structure']}
- Front month: ${metrics['front_price']:.2f}
- Back month: ${metrics['back_price']:.2f}
- Front-back spread: ${metrics['front_back_spread']:.2f}
- Annualized slope: {metrics['slope_annualized']*100:.2f}%

RECENT NEWS CONTEXT:
{news_str}

Provide comprehensive analysis as JSON:
{{
  "structure_interpretation": {{
    "current_state": "Describe what the curve shape indicates",
    "strength": "How strong is the contango/backwardation?",
    "historical_context": "Is this normal or extreme for this commodity?"
  }},
  "fundamental_drivers": [
    "List key factors driving current term structure",
    "Example: High inventories supporting contango",
    "Example: OPEC cuts creating backwardation"
  ],
  "forward_outlook": {{
    "expected_direction": "steepening | flattening | inverting",
    "timeframe": "Expected timeframe for change",
    "reasoning": "Why will the curve change (or stay stable)?",
    "catalysts": ["Specific events that could shift structure"]
  }},
  "trading_implications": {{
    "calendar_spreads": "Which spreads offer opportunities?",
    "roll_yield": "Positive or negative for long positions?",
    "positioning": "Should traders be long front or back?"
  }},
  "risk_factors": [
    "What could invalidate this analysis?",
    "Unexpected events to monitor"
  ]
}}

Important:
- Consider supply/demand balance from news
- Explain calendar spread patterns
- Assess whether curve reflects fundamentals or positioning"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        return json.loads(response.content[0].text)


# Example: Crude oil term structure analysis
analyzer = TermStructureAnalyzer(anthropic_api_key="your-key")

# Generate synthetic futures curve
contracts = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M12']
days_to_expiry = [30, 60, 90, 120, 150, 180, 365]

# Contango curve
contango_prices = [70 + i*0.5 for i in range(len(contracts))]

# Backwardation curve
backwardation_prices = [80 - i*0.3 for i in range(len(contracts))]

# Create DataFrame
futures_data_contango = pd.DataFrame({
    'date': '2024-01-15',
    'contract': contracts,
    'price': contango_prices,
    'maturity': [datetime(2024, 1, 15) + timedelta(days=d) for d in days_to_expiry]
})

futures_data_backwardation = pd.DataFrame({
    'date': '2024-06-15',
    'contract': contracts,
    'price': backwardation_prices,
    'maturity': [datetime(2024, 6, 15) + timedelta(days=d) for d in days_to_expiry]
})

# Analyze contango
curve_contango = analyzer.extract_curve(futures_data_contango, '2024-01-15')
metrics_contango = analyzer.compute_metrics(curve_contango)

print("CONTANGO STRUCTURE:")
print(f"  Front: ${metrics_contango['front_price']:.2f}")
print(f"  Back: ${metrics_contango['back_price']:.2f}")
print(f"  Spread: ${metrics_contango['front_back_spread']:.2f}")
print(f"  Slope: {metrics_contango['slope_annualized']*100:.2f}%")

fig1 = analyzer.visualize_curve(curve_contango, metrics_contango,
                                title='Crude Oil Term Structure - Contango')
plt.savefig('term_structure_contango.png', dpi=150, bbox_inches='tight')

# Analyze backwardation
curve_back = analyzer.extract_curve(futures_data_backwardation, '2024-06-15')
metrics_back = analyzer.compute_metrics(curve_back)

print("\nBACKWARDATION STRUCTURE:")
print(f"  Front: ${metrics_back['front_price']:.2f}")
print(f"  Back: ${metrics_back['back_price']:.2f}")
print(f"  Spread: ${metrics_back['front_back_spread']:.2f}")
print(f"  Slope: {metrics_back['slope_annualized']*100:.2f}%")

fig2 = analyzer.visualize_curve(curve_back, metrics_back,
                               title='Crude Oil Term Structure - Backwardation')
plt.savefig('term_structure_backwardation.png', dpi=150, bbox_inches='tight')

# LLM interpretation
news = [
    "OPEC announces surprise 1 million bpd production cut effective immediately",
    "U.S. crude inventories fall 5 million barrels, largest drop in 6 months",
    "Refinery utilization rates climb to 95%, highest since 2019"
]

interpretation = analyzer.llm_interpret_structure(curve_back, metrics_back, news)

print("\nLLM INTERPRETATION:")
print(json.dumps(interpretation, indent=2))

plt.show()
```

</div>
</div>

### Curve Evolution Analysis


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">analyze_curve_evolution.py</span>
</div>

```python
def analyze_curve_evolution(futures_data, dates):
    """
    Track how term structure evolves over time.

    Args:
        futures_data: Full futures dataset
        dates: List of analysis dates

    Returns:
        evolution: Time series of term structure metrics
    """
    analyzer = TermStructureAnalyzer(anthropic_api_key="your-key")

    evolution = []

    for date in dates:
        curve = analyzer.extract_curve(futures_data, date)
        metrics = analyzer.compute_metrics(curve)

        evolution.append({
            'date': date,
            'structure': metrics['structure'],
            'front_price': metrics['front_price'],
            'spread': metrics['front_back_spread'],
            'slope': metrics['slope_annualized']
        })

    df = pd.DataFrame(evolution)

    # Visualize evolution
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Front month price
    axes[0].plot(df['date'], df['front_price'], linewidth=2)
    axes[0].set_ylabel('Front Month Price ($)')
    axes[0].set_title('Front Month Price Evolution')
    axes[0].grid(alpha=0.3)

    # Plot 2: Front-back spread
    axes[1].plot(df['date'], df['spread'], linewidth=2, color='blue')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].fill_between(df['date'], 0, df['spread'],
                        where=(df['spread'] > 0),
                        alpha=0.3, color='green', label='Contango')
    axes[1].fill_between(df['date'], 0, df['spread'],
                        where=(df['spread'] < 0),
                        alpha=0.3, color='red', label='Backwardation')
    axes[1].set_ylabel('Front-Back Spread ($)')
    axes[1].set_title('Term Structure Evolution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot 3: Slope
    axes[2].plot(df['date'], df['slope']*100, linewidth=2, color='purple')
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Annualized Slope (%)')
    axes[2].set_title('Curve Slope Over Time')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return df, fig
```

</div>
</div>

## Common Pitfalls

**1. Ignoring Roll Costs**
- **Problem:** Trading front-month contract without considering negative roll yield in contango
- **Symptom:** Unexpected losses despite correct price direction
- **Solution:** Calculate roll yield, consider calendar spread trades

**2. Mis-Interpreting Curve Shape**
- **Problem:** Assuming backwardation always means bullish
- **Symptom:** Wrong positioning when backwardation reflects expected supply increase
- **Solution:** LLM contextual analysis—why is curve shaped this way?

**3. Static Analysis**
- **Problem:** Analyzing curve at single point in time
- **Symptom:** Missing structural shifts (contango → backwardation transition)
- **Solution:** Track curve evolution, monitor slope changes

**4. Ignoring Seasonality**
- **Problem:** Comparing term structures across seasons without adjustment
- **Symptom**: Natural gas contango in summer seems "expensive" but is normal
- **Solution:** Compare to historical seasonal patterns

**5. Confusing Spread Direction**
- **Problem:** Long M2-M1 spread expecting backwardation to widen (but M2>M1 in contango!)
- **Symptom:** Position loses money despite correct fundamental view
- **Solution:** Clearly define spread convention, verify backwardation/contango

## Connections

**Builds on:**
- Module 4.1: Supply/demand fundamentals (drive curve shape)
- Module 4.2: Storage analysis (storage cost affects contango)
- Options pricing (term structure affects vol surface)

**Leads to:**
- Trading strategies (calendar spreads, roll optimization)
- Risk management (curve risk, spread risk)
- Portfolio construction (front vs back positioning)

**Related concepts:**
- Interest rate term structure (yield curve)
- Volatility term structure (VIX curve)
- Credit curves (CDS term structure)

## Practice Problems

1. **Spread Calculation**
   Futures prices:
   - M1: $70
   - M2: $71.50
   - M3: $72.80

   Calculate:
   - M1-M2 spread
   - M2-M3 spread
   - Is this contango or backwardation?
   - Is the curve steepening or flattening from M1 to M3?

2. **Roll Yield**
   Front month contract: $75
   Back month contract: $78 (1 month later)

   What's the roll yield for a long position?
   If spot price doesn't change, what's your P&L from rolling?

3. **Storage Arbitrage**
   Spot price: $70
   3-month futures: $74
   Storage cost: $0.50/barrel/month
   Risk-free rate: 5% annual

   Is there an arbitrage opportunity?

4. **LLM Prompt Design**
   Design a prompt for LLM to predict whether crude oil curve will steepen or flatten given:
   - Current backwardation of $3 (front - back)
   - News: "OPEC to increase production next quarter"
   - Inventory: 5% above 5-year average

5. **Calendar Spread Trade**
   You expect backwardation to widen.
   Current: M1=$80, M2=$78 (backwardation of $2)
   Expected: M1=$82, M2=$78 (backwardation of $4)

   What position should you take? (Long M1-M2 or short M1-M2?)

<div class="callout-insight">

**Insight:** Understanding term structure analysis with llms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Term Structure Theory:**
<div class="flow">
<div class="flow-step mint">1. Fama & French (1987)</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Gorton et al. (2013)</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Pindyck (2001)</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Szymanowska et al. (...</div>


1. **Fama & French (1987)** - "Commodity Futures Prices" - Theory of storage
2. **Gorton et al. (2013)** - "The Fundamentals of Commodity Futures Returns" - Roll yield analysis
3. **Pindyck (2001)** - "The Dynamics of Commodity Spot and Futures Markets" - Curve dynamics

**Empirical Studies:**
4. **Szymanowska et al. (2014)** - "An Anatomy of Commodity Futures Risk Premia" - Term structure risk
5. **Hamilton & Wu (2015)** - "Risk Premia in Crude Oil Futures Prices" - Oil curve analysis

**Trading Applications:**
6. **Till (2006)** - "A Long-Term Perspective on Commodity Futures Returns" - Roll yield strategies
7. **Erb & Harvey (2006)** - "The Tactical and Strategic Value of Commodity Futures" - Curve positioning

**Natural Gas Specific:**
8. **Pirrong (2012)** - "Commodity Price Dynamics" - Natural gas seasonality and storage

**LLM Applications:**
9. **"Using LLMs for Market Structure Analysis"** - Curve interpretation
10. **"Augmenting Quantitative Strategies with Narrative Analysis"** - Combining curves with news

*"The futures curve is a market's expectation written in prices. LLMs read the news to explain why the market thinks that way."*

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./03_term_structure_slides.md">
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

