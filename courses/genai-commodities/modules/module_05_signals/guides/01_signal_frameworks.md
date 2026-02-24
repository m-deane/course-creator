# Signal Generation Frameworks with LLMs

## In Brief

Signal generation frameworks systematically transform LLM outputs (sentiment, events, fundamental analysis) into actionable trading signals with quantified conviction levels. These frameworks combine narrative intelligence with quantitative rigor, converting qualitative insights into systematic position sizing, risk management, and portfolio decisions.

> 💡 **Key Insight:** LLMs excel at understanding context but don't naturally output tradeable signals. A signal framework bridges this gap by: (1) structuring LLM outputs into standardized formats, (2) mapping narrative conviction to position sizes, (3) combining multiple signals with conflict resolution, (4) tracking signal performance for continuous improvement. The goal: transform "OPEC cuts seem bullish" into "Long 100 contracts WTI with 2% stop loss."

## Formal Definition

### Signal Structure

**A trading signal is a tuple:**
$$\text{Signal} = (A, D, S, C, H, R)$$

Where:
- $A$: Action ∈ {Long, Short, Flat}
- $D$: Direction ∈ {Bullish, Bearish, Neutral}
- $S$: Strength ∈ [0, 1] (conviction level)
- $C$: Catalyst (event or condition triggering signal)
- $H$: Time horizon ∈ {Intraday, Short-term, Medium-term, Long-term}
- $R$: Risk parameters {stop_loss, take_profit, max_position}

### Signal Generation Pipeline

```
Raw Information → LLM Analysis → Structured Output → Signal Transformation → Position Sizing
```

**Step 1: LLM Analysis**
Input: News, fundamental data, technical levels
Output: Narrative assessment with conviction

**Step 2: Signal Extraction**
```python
{
  "direction": "bullish",
  "conviction": 0.75,
  "catalyst": "OPEC production cut",
  "horizon": "medium-term",
  "reasoning": "Supply reduction with stable demand"
}
```

**Step 3: Position Sizing**
$$\text{Position Size} = \text{Base Size} \times S \times \min(1, \text{Kelly Fraction})$$

**Step 4: Risk Management**
- Stop loss: $\text{SL} = \text{Entry} \times (1 - \alpha \cdot S)$ for long
- Take profit: $\text{TP} = \text{Entry} \times (1 + \beta \cdot S)$ for long

### Multi-Signal Aggregation

Given N signals $\{(A_i, D_i, S_i)\}_{i=1}^N$:

**Weighted average direction:**
$$D_{\text{agg}} = \frac{\sum_{i=1}^N S_i \cdot \text{sign}(D_i)}{\sum_{i=1}^N S_i}$$

Where sign(Bullish) = +1, sign(Bearish) = -1

**Conflict detection:**
If $|D_{\text{agg}}| < \theta$ (threshold), signals conflict → reduce position or stay flat

## Intuitive Explanation

Think of a trading team with multiple analysts:

**Without framework:**
- Analyst A: "I think oil is bullish"
- Analyst B: "Oil looks bearish to me"
- Analyst C: "Maybe neutral?"
- Trader: Confused, no position taken

**With signal framework:**
- Analyst A: Bullish, 75% conviction, 1-month horizon (Supply cut)
- Analyst B: Bearish, 60% conviction, 1-week horizon (Short-term oversupply)
- Analyst C: Neutral, 50% conviction, long-term (Uncertain fundamentals)

**Framework resolution:**
1. Separate by horizon: A (1-month) vs B (1-week) → Compatible
2. Weight by conviction: A (75%) > B (60%) → Net bullish tilt
3. Position: Small long (conflicting signals reduce size)
4. Risk: Tight stop (uncertainty present)

For LLM signals:

**LLM 1 (News analysis):**
"OPEC cut is significantly bullish, convoy 80%"

**LLM 2 (Fundamental analysis):**
"Inventory levels suggest bearish, conviction 65%"

**Framework aggregation:**
- Net signal: 0.8 × 1 + 0.65 × (-1) = 0.15 (slightly bullish)
- Action: Small long position (conflicting signals)
- Stop: Tight (protect against LLM 2 being correct)

## Code Implementation

### Core Signal Framework

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from anthropic import Anthropic

class Direction(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

class Action(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class Horizon(Enum):
    INTRADAY = "intraday"
    SHORT_TERM = "short_term"  # Days
    MEDIUM_TERM = "medium_term"  # Weeks
    LONG_TERM = "long_term"  # Months

@dataclass
class TradingSignal:
    """
    Structured trading signal from LLM analysis.
    """
    action: Action
    direction: Direction
    strength: float  # [0, 1] conviction level
    catalyst: str
    horizon: Horizon
    reasoning: str

    # Risk parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position_pct: float = 1.0  # % of typical position size

    # Metadata
    source: str = "LLM"
    timestamp: str = None
    confidence_factors: Dict = None

    def __post_init__(self):
        # Validate strength
        if not 0 <= self.strength <= 1:
            raise ValueError("Strength must be in [0, 1]")

        # Set default timestamp
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()


class SignalGenerator:
    """
    Generate trading signals from LLM analysis.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def generate_signal(
        self,
        context: str,
        current_price: float,
        asset: str = "crude_oil"
    ) -> TradingSignal:
        """
        Generate trading signal from contextual information.

        Args:
            context: News, fundamentals, technical levels
            current_price: Current asset price
            asset: Asset identifier

        Returns:
            TradingSignal object
        """
        prompt = f"""You are a commodity trading analyst. Analyze this information and generate a trading signal.

ASSET: {asset}
CURRENT PRICE: ${current_price:.2f}

CONTEXT:
{context}

Generate a trading signal as JSON with the following structure:
{{
  "direction": "bullish" | "bearish" | "neutral",
  "strength": 0.0-1.0,
  "catalyst": "Primary factor driving this signal",
  "horizon": "intraday" | "short_term" | "medium_term" | "long_term",
  "reasoning": "Detailed explanation of signal logic",
  "confidence_factors": {{
    "fundamental_support": 0.0-1.0,
    "technical_confirmation": 0.0-1.0,
    "sentiment_alignment": 0.0-1.0,
    "timing": 0.0-1.0
  }},
  "risk_parameters": {{
    "stop_loss_pct": "% below/above entry",
    "take_profit_pct": "% above/below entry",
    "max_position_pct": "% of normal position size"
  }}
}}

Guidelines:
- Strength: 0.0-0.3 (weak), 0.3-0.6 (moderate), 0.6-0.8 (strong), 0.8-1.0 (very strong)
- Consider multiple factors: fundamentals, technicals, sentiment, timing
- Be specific about catalyst (not just "bullish outlook")
- Adjust stop loss tighter for lower conviction signals
- Conflicting factors should reduce strength"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1536,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        import json
        signal_data = json.loads(response.content[0].text)

        # Map to TradingSignal
        direction_map = {
            "bullish": Direction.BULLISH,
            "bearish": Direction.BEARISH,
            "neutral": Direction.NEUTRAL
        }

        horizon_map = {
            "intraday": Horizon.INTRADAY,
            "short_term": Horizon.SHORT_TERM,
            "medium_term": Horizon.MEDIUM_TERM,
            "long_term": Horizon.LONG_TERM
        }

        direction = direction_map[signal_data['direction']]

        # Determine action from direction
        if direction == Direction.BULLISH:
            action = Action.LONG
        elif direction == Direction.BEARISH:
            action = Action.SHORT
        else:
            action = Action.FLAT

        # Calculate risk parameters
        risk_params = signal_data['risk_parameters']
        stop_loss_pct = risk_params['stop_loss_pct']
        take_profit_pct = risk_params['take_profit_pct']

        if action == Action.LONG:
            stop_loss = current_price * (1 - stop_loss_pct / 100)
            take_profit = current_price * (1 + take_profit_pct / 100)
        elif action == Action.SHORT:
            stop_loss = current_price * (1 + stop_loss_pct / 100)
            take_profit = current_price * (1 - take_profit_pct / 100)
        else:
            stop_loss = None
            take_profit = None

        signal = TradingSignal(
            action=action,
            direction=direction,
            strength=signal_data['strength'],
            catalyst=signal_data['catalyst'],
            horizon=horizon_map[signal_data['horizon']],
            reasoning=signal_data['reasoning'],
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position_pct=risk_params['max_position_pct'],
            confidence_factors=signal_data['confidence_factors'],
            source="LLM"
        )

        return signal


class SignalAggregator:
    """
    Aggregate multiple signals with conflict resolution.
    """

    def aggregate(self, signals: List[TradingSignal]) -> TradingSignal:
        """
        Combine multiple signals into single aggregate signal.

        Uses conviction-weighted averaging with conflict detection.
        """
        if not signals:
            raise ValueError("No signals to aggregate")

        # Filter by horizon (only aggregate similar horizons)
        # For simplicity, use all signals; production would separate by horizon

        # Compute weighted direction
        total_weight = sum(s.strength for s in signals)
        weighted_direction = sum(
            s.strength * s.direction.value for s in signals
        ) / total_weight

        # Detect conflicts
        conflict_threshold = 0.3
        has_conflict = abs(weighted_direction) < conflict_threshold

        # Determine aggregate action
        if has_conflict:
            action = Action.FLAT
            direction = Direction.NEUTRAL
            strength = 0.0
        elif weighted_direction > 0:
            action = Action.LONG
            direction = Direction.BULLISH
            strength = min(weighted_direction, 1.0)
        else:
            action = Action.SHORT
            direction = Direction.BEARISH
            strength = min(abs(weighted_direction), 1.0)

        # Reduce strength if conflict present
        conflict_penalty = 0.7 if has_conflict else 1.0
        strength *= conflict_penalty

        # Combine reasoning
        reasoning_parts = [f"Signal {i+1}: {s.reasoning}" for i, s in enumerate(signals)]
        combined_reasoning = "\n\n".join(reasoning_parts)
        combined_reasoning += f"\n\nAGGREGATE: Weighted direction = {weighted_direction:.2f}"

        # Combine catalysts
        catalysts = [s.catalyst for s in signals]
        combined_catalyst = " | ".join(set(catalysts))

        # Use most common horizon
        from collections import Counter
        horizon_counts = Counter(s.horizon for s in signals)
        aggregate_horizon = horizon_counts.most_common(1)[0][0]

        # Average entry price
        avg_entry = np.mean([s.entry_price for s in signals if s.entry_price is not None])

        # Conservative risk (use tightest stop)
        if action == Action.LONG:
            stop_losses = [s.stop_loss for s in signals if s.stop_loss is not None]
            aggregate_stop = max(stop_losses) if stop_losses else None
        elif action == Action.SHORT:
            stop_losses = [s.stop_loss for s in signals if s.stop_loss is not None]
            aggregate_stop = min(stop_losses) if stop_losses else None
        else:
            aggregate_stop = None

        aggregate_signal = TradingSignal(
            action=action,
            direction=direction,
            strength=strength,
            catalyst=combined_catalyst,
            horizon=aggregate_horizon,
            reasoning=combined_reasoning,
            entry_price=avg_entry,
            stop_loss=aggregate_stop,
            take_profit=None,  # Recompute based on aggregate
            source="Aggregated"
        )

        return aggregate_signal


# Example usage
generator = SignalGenerator(anthropic_api_key="your-key")

# Scenario 1: Bullish news
context1 = """
OPEC+ announces surprise production cut of 1 million barrels per day starting next month.
Saudi Arabia extends voluntary cut through Q4 2024.
U.S. crude inventories fell 5.2 million barrels last week, largest draw in 6 months.
Brent-WTI spread widened to $5, indicating tight U.S. supply.
"""

signal1 = generator.generate_signal(context1, current_price=75.50, asset="WTI Crude")

print("=" * 60)
print("SIGNAL 1: OPEC CUT NEWS")
print("=" * 60)
print(f"Action: {signal1.action.value}")
print(f"Direction: {signal1.direction.name}")
print(f"Strength: {signal1.strength:.2f}")
print(f"Catalyst: {signal1.catalyst}")
print(f"Horizon: {signal1.horizon.value}")
print(f"Entry: ${signal1.entry_price:.2f}")
print(f"Stop Loss: ${signal1.stop_loss:.2f}" if signal1.stop_loss else "Stop Loss: N/A")
print(f"Take Profit: ${signal1.take_profit:.2f}" if signal1.take_profit else "Take Profit: N/A")
print(f"\nReasoning:\n{signal1.reasoning}")
print(f"\nConfidence Factors:")
for factor, value in signal1.confidence_factors.items():
    print(f"  {factor}: {value:.2f}")

# Scenario 2: Conflicting signals
context2 = """
Chinese PMI data comes in below expectations, suggesting weaker demand.
However, refinery margins (crack spreads) remain elevated, indicating strong processing demand.
Technical analysis shows price at key resistance level.
"""

signal2 = generator.generate_signal(context2, current_price=75.50, asset="WTI Crude")

print("\n" + "=" * 60)
print("SIGNAL 2: CONFLICTING FACTORS")
print("=" * 60)
print(f"Action: {signal2.action.value}")
print(f"Strength: {signal2.strength:.2f}")
print(f"Catalyst: {signal2.catalyst}")

# Aggregate signals
aggregator = SignalAggregator()
aggregate = aggregator.aggregate([signal1, signal2])

print("\n" + "=" * 60)
print("AGGREGATED SIGNAL")
print("=" * 60)
print(f"Action: {aggregate.action.value}")
print(f"Strength: {aggregate.strength:.2f}")
print(f"Catalyst: {aggregate.catalyst}")
print(f"\nReasoning:\n{aggregate.reasoning}")
```

### Position Sizing from Signals

```python
class PositionSizer:
    """
    Convert signals to position sizes with risk management.
    """

    def __init__(
        self,
        account_value: float,
        max_risk_per_trade: float = 0.02,  # 2% max risk
        base_position_pct: float = 0.10  # 10% of account per position
    ):
        self.account_value = account_value
        self.max_risk_per_trade = max_risk_per_trade
        self.base_position_pct = base_position_pct

    def size_position(self, signal: TradingSignal) -> Dict:
        """
        Calculate position size from signal.

        Returns:
            position_details: Dict with size, risk, expectation
        """
        if signal.action == Action.FLAT:
            return {
                'contracts': 0,
                'dollar_size': 0,
                'risk_amount': 0,
                'position_pct': 0
            }

        # Base position size
        base_dollar_size = self.account_value * self.base_position_pct

        # Adjust by signal strength
        adjusted_size = base_dollar_size * signal.strength

        # Adjust by max position constraint from signal
        adjusted_size *= signal.max_position_pct

        # Calculate risk per contract
        if signal.stop_loss:
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        else:
            # Default risk: 2% of price
            risk_per_unit = signal.entry_price * 0.02

        # Maximum contracts based on risk limit
        max_risk_dollars = self.account_value * self.max_risk_per_trade
        max_contracts_by_risk = max_risk_dollars / risk_per_unit

        # Calculate contracts (assuming 1 contract = 1 barrel for simplicity)
        contracts_by_size = adjusted_size / signal.entry_price
        contracts = min(contracts_by_size, max_contracts_by_risk)

        # Actual risk
        risk_amount = contracts * risk_per_unit

        return {
            'contracts': int(contracts),
            'dollar_size': contracts * signal.entry_price,
            'risk_amount': risk_amount,
            'risk_pct': risk_amount / self.account_value,
            'position_pct': (contracts * signal.entry_price) / self.account_value
        }


# Example position sizing
sizer = PositionSizer(account_value=1_000_000, max_risk_per_trade=0.02)

position1 = sizer.size_position(signal1)
print("\nPOSITION SIZING (Strong Signal):")
print(f"  Contracts: {position1['contracts']}")
print(f"  Dollar Size: ${position1['dollar_size']:,.0f}")
print(f"  Risk Amount: ${position1['risk_amount']:,.0f}")
print(f"  Risk %: {position1['risk_pct']*100:.2f}%")
print(f"  Position %: {position1['position_pct']*100:.2f}%")

position2 = sizer.size_position(signal2)
print("\nPOSITION SIZING (Conflicting Signal):")
print(f"  Contracts: {position2['contracts']}")
print(f"  Dollar Size: ${position2['dollar_size']:,.0f}")
```

## Common Pitfalls

**1. Over-Trusting LLM Conviction**
- **Problem:** LLM says "very strong bullish" without proper context
- **Symptom:** Over-sized positions on weak fundamentals
- **Solution:** Calibrate LLM outputs against historical performance

**2. Ignoring Signal Conflicts**
- **Problem:** Taking signals in isolation without checking agreement
- **Symptom:** Whipsawed by contradictory signals
- **Solution:** Implement conflict detection, reduce size when signals disagree

**3. No Time Horizon Separation**
- **Problem:** Mixing intraday signals with monthly signals
- **Symptom:** Confusion, premature exits
- **Solution:** Separate signal pipelines by horizon, aggregate only within horizon

**4. Static Risk Parameters**
- **Problem:** Same stop loss % for all signals
- **Symptom:** Inappropriate risk for conviction level
- **Solution:** Dynamically adjust risk based on signal strength and confidence factors

**5. No Signal Performance Tracking**
- **Problem:** Not measuring if LLM signals actually work
- **Symptom:** Persistent losses despite "strong signals"
- **Solution:** Log all signals, track P&L attribution, iterate prompts

## Connections

**Builds on:**
- Module 4: Fundamental analysis (input to signal generation)
- Module 3: Sentiment analysis (another signal source)
- Portfolio theory (position sizing, risk management)

**Leads to:**
- Module 5.2: Confidence scoring (quantifying signal reliability)
- Module 5.3: Backtesting (validating signal performance)
- Module 6: Production deployment (systematic signal execution)

**Related frameworks:**
- Alpha generation (signals as alpha sources)
- Risk parity (balancing signal contributions)
- Multi-factor models (combining fundamental, technical, sentiment signals)

## Practice Problems

1. **Signal Strength Calibration**
   LLM outputs:
   - Signal A: "Strong bullish" (0.9 conviction)
   - Signal B: "Moderate bullish" (0.6 conviction)

   Historical win rates: A = 55%, B = 53%

   Should conviction track win rate? How to calibrate?

2. **Conflict Resolution**
   Three signals:
   - Technical: Bullish, 0.7 strength
   - Fundamental: Bearish, 0.8 strength
   - Sentiment: Neutral, 0.5 strength

   Calculate weighted direction. What action?

3. **Position Sizing**
   - Account: $1M
   - Signal: Bullish, 0.75 strength
   - Entry: $75
   - Stop: $72
   - Max risk per trade: 2%

   Calculate: Max contracts by risk? Position size?

4. **Horizon Mismatch**
   - Signal A: Bullish, short-term (1 week)
   - Signal B: Bearish, long-term (3 months)

   How to position? Can both be correct simultaneously?

5. **LLM Prompt Engineering**
   Design a prompt that generates signals with:
   - Explicit confidence intervals (not just point estimate)
   - Multiple scenarios (base/bull/bear cases)
   - Time-conditional triggers ("if inventory drops below X, increase conviction")

## Further Reading

**Signal Generation:**
1. **"Systematic Trading" by Robert Carver** - Signal generation frameworks
2. **"Quantitative Trading" by Ernest Chan** - Alpha generation

**Position Sizing:**
3. **"The Mathematics of Money Management" by Ralph Vince** - Kelly criterion, optimal f
4. **"Trade Your Way to Financial Freedom" by Van Tharp** - Position sizing strategies

**Risk Management:**
5. **"Options, Futures, and Other Derivatives" by Hull** - Hedging signals
6. **"Risk Management and Financial Institutions" by Hull** - Portfolio risk

**LLM Applications:**
7. **"LLMs for Systematic Trading Signals"** (2024) - Recent research
8. **"Augmenting Alpha Generation with Language Models"** - Practical applications

**Commodity-Specific:**
9. **"Inside the House of Money" by Steven Drobny** - Macro signal frameworks
10. **"The New Commodity Trading Guide" by Kleinman** - Signal implementation

---

*"Signals are where narrative meets numbers. A framework ensures LLM insights become systematic, risk-managed trades."*
