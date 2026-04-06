# Building Trading Signals from Commodity Sentiment

> **Reading time:** ~13 min | **Module:** Module 3: Sentiment | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Signal construction transforms sentiment analysis into actionable trading recommendations by combining sentiment strength, confidence levels, historical validation, and risk management rules to produce position sizing, entry/exit timing, and stop-loss levels for commodity trades.

</div>

## In Brief

Signal construction transforms sentiment analysis into actionable trading recommendations by combining sentiment strength, confidence levels, historical validation, and risk management rules to produce position sizing, entry/exit timing, and stop-loss levels for commodity trades.

<div class="callout-insight">

**Insight:** Sentiment is not a signal—it's a signal ingredient. "Bullish sentiment" alone doesn't tell you whether to buy now, how much to buy, or when to exit. Effective signal construction requires combining sentiment with price action confirmation, position sizing based on confidence, and historical backtesting to validate that sentiment actually predicts price moves.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of building trading signals like a pilot's pre-flight checklist:

**Bad approach (sentiment only):**
- Pilot sees "weather looks good" → takes off immediately
- No check of fuel, instruments, or flight path
- Result: Dangerous

**Good approach (complete signal):**
- Pilot sees "weather looks good" ✓
- Checks fuel levels ✓
- Verifies instruments working ✓
- Confirms flight path clear ✓
- Assesses risk (emergency landing sites) ✓
- **Then** takes off

**Trading signal equivalent:**
- Sentiment is bullish ✓ (weather good)
- Price confirms (breaking resistance) ✓ (instruments working)
- Volatility acceptable ✓ (safe conditions)
- Position size calculated ✓ (fuel sufficient)
- Stop-loss set ✓ (emergency plan)
- **Then** enter trade

The signal combines multiple factors into a go/no-go decision with specific parameters.

## Formal Definition

A trading signal is a function **SIG: (Sent, Price, Risk) → Action** where:

**Inputs:**
- **Sent** = sentiment analysis {direction, confidence, key_factors}
- **Price** = market data {current_price, recent_trend, volatility}
- **Risk** = risk parameters {max_position_size, stop_loss_pct, target_profit}

**Output:** Action = trading recommendation:
```
Action = {
  signal_type: {long | short | flat | reduce},
  strength: [0, 1],  # Position sizing factor
  entry_price: target_entry,
  stop_loss: price level to exit if wrong,
  take_profit: price level to exit if right,
  time_horizon: expected holding period,
  confidence: [0, 1],
  rationale: explanation of signal
}
```

**Signal Construction Rules:**
<div class="flow">
<div class="flow-step mint">1. Sentiment Threshold</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Confirmation</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Position Sizing</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Risk Management</div>
</div>


1. **Sentiment Threshold**: Only act on high-confidence sentiment (>0.7)
2. **Confirmation**: Require price action confirmation (don't fight strong trends)
3. **Position Sizing**: Scale position by sentiment strength × confidence
4. **Risk Management**: Stop-loss based on recent volatility (e.g., 2× ATR)
5. **Time Decay**: Sentiment signals decay over time (hourly for intraday, daily for swing trades)

## Code Implementation

### Signal Generator from Sentiment


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

class SignalType(Enum):
    LONG = "long"  # Buy/bullish position
    SHORT = "short"  # Sell/bearish position
    FLAT = "flat"  # No position
    REDUCE = "reduce"  # Reduce existing position
    ADD = "add"  # Add to existing position

class TimeHorizon(Enum):
    INTRADAY = "intraday"  # Hours
    SWING = "swing"  # Days to weeks
    POSITION = "position"  # Weeks to months

@dataclass
class MarketData:
    """Current market state."""
    current_price: float
    daily_high: float
    daily_low: float
    volume: float
    atr_20: float  # 20-period Average True Range (volatility)
    sma_50: float  # 50-day simple moving average
    sma_200: float  # 200-day simple moving average
    rsi_14: float  # 14-period Relative Strength Index

@dataclass
class TradingSignal:
    """Complete trading signal with entry/exit parameters."""
    signal_type: SignalType
    strength: float  # 0-1, position sizing factor
    entry_price: float
    stop_loss: float
    take_profit: float
    time_horizon: TimeHorizon
    confidence: float
    rationale: str
    generated_at: datetime

    # Risk metrics
    risk_reward_ratio: float
    max_loss_pct: float
    expected_gain_pct: float

class SignalConstructor:
    """
    Construct trading signals from sentiment and market data.
    """

    def __init__(
        self,
        min_sentiment_confidence: float = 0.7,
        min_signal_strength: float = 0.5,
        default_risk_reward: float = 2.0,  # Target 2:1 reward:risk
        max_position_pct: float = 0.1  # Max 10% of portfolio
    ):
        self.min_sentiment_confidence = min_sentiment_confidence
        self.min_signal_strength = min_signal_strength
        self.default_risk_reward = default_risk_reward
        self.max_position_pct = max_position_pct

    def construct_signal(
        self,
        sentiment: 'CommoditySentiment',
        market_data: MarketData,
        current_position: Optional[float] = None
    ) -> Optional[TradingSignal]:
        """
        Construct trading signal from sentiment and market data.

        Args:
            sentiment: Extracted sentiment analysis
            market_data: Current market state
            current_position: Current position size (if any)

        Returns:
            TradingSignal or None if no signal
        """
        # Step 1: Check sentiment confidence threshold
        if sentiment.confidence < self.min_sentiment_confidence:
            return None

        # Step 2: Determine base signal direction
        if sentiment.overall_direction == SentimentDirection.BULLISH:
            base_direction = SignalType.LONG
        elif sentiment.overall_direction == SentimentDirection.BEARISH:
            base_direction = SignalType.SHORT
        else:
            return None  # No signal on neutral/mixed sentiment

        # Step 3: Check price confirmation
        price_confirms = self._check_price_confirmation(
            sentiment.overall_direction,
            market_data
        )

        if not price_confirms:
            # Sentiment and price disagree - no signal
            return None

        # Step 4: Calculate signal strength
        strength = self._calculate_signal_strength(sentiment, market_data)

        if strength < self.min_signal_strength:
            return None

        # Step 5: Determine entry, stop, and target
        entry, stop, target = self._calculate_levels(
            base_direction,
            market_data,
            sentiment
        )

        # Step 6: Calculate risk metrics
        risk_reward = abs(target - entry) / abs(entry - stop)
        max_loss_pct = abs(entry - stop) / entry
        expected_gain_pct = abs(target - entry) / entry

        # Step 7: Build rationale
        rationale = self._build_rationale(sentiment, market_data, base_direction)

        # Step 8: Determine time horizon
        time_horizon = self._determine_time_horizon(sentiment)

        return TradingSignal(
            signal_type=base_direction,
            strength=strength,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            time_horizon=time_horizon,
            confidence=sentiment.confidence,
            rationale=rationale,
            generated_at=datetime.now(),
            risk_reward_ratio=risk_reward,
            max_loss_pct=max_loss_pct,
            expected_gain_pct=expected_gain_pct
        )

    def _check_price_confirmation(
        self,
        sentiment_direction: SentimentDirection,
        market_data: MarketData
    ) -> bool:
        """
        Check if price action confirms sentiment.

        Confirmation rules:
        - Bullish: Price above SMA50 or recent breakout
        - Bearish: Price below SMA50 or recent breakdown
        - Don't fight strong opposite trends
        """
        price = market_data.current_price

        if sentiment_direction == SentimentDirection.BULLISH:
            # Bullish confirmation
            if price > market_data.sma_50:
                return True
            # Or recent breakout (price near daily high)
            if price > market_data.daily_high * 0.98:
                return True
            # Don't go long in strong downtrend
            if price < market_data.sma_200 * 0.95:
                return False

        elif sentiment_direction == SentimentDirection.BEARISH:
            # Bearish confirmation
            if price < market_data.sma_50:
                return True
            # Or recent breakdown
            if price < market_data.daily_low * 1.02:
                return True
            # Don't go short in strong uptrend
            if price > market_data.sma_200 * 1.05:
                return False

        return True  # Default to confirmed

    def _calculate_signal_strength(
        self,
        sentiment: 'CommoditySentiment',
        market_data: MarketData
    ) -> float:
        """
        Calculate signal strength for position sizing.

        Factors:
        - Sentiment confidence (40%)
        - Number of bullish/bearish factors (30%)
        - Price momentum alignment (20%)
        - Low volatility environment (10% - prefer low vol)
        """
        strength = 0.0

        # 1. Sentiment confidence (40%)
        strength += sentiment.confidence * 0.4

        # 2. Number of key factors (30%)
        bullish_factors = len(sentiment.key_factors.get('bullish', []))
        bearish_factors = len(sentiment.key_factors.get('bearish', []))

        if sentiment.overall_direction == SentimentDirection.BULLISH:
            factor_score = min(1.0, bullish_factors / 5)  # 5+ factors = max
        else:
            factor_score = min(1.0, bearish_factors / 5)

        strength += factor_score * 0.3

        # 3. Price momentum alignment (20%)
        rsi = market_data.rsi_14
        if sentiment.overall_direction == SentimentDirection.BULLISH:
            # Bullish but not overbought
            momentum_score = 1.0 if 40 < rsi < 70 else 0.5
        else:
            # Bearish but not oversold
            momentum_score = 1.0 if 30 < rsi < 60 else 0.5

        strength += momentum_score * 0.2

        # 4. Volatility factor (10%)
        # Prefer lower volatility for clearer signals
        # ATR as % of price
        volatility_pct = market_data.atr_20 / market_data.current_price

        if volatility_pct < 0.02:  # Low volatility
            vol_score = 1.0
        elif volatility_pct < 0.05:  # Moderate
            vol_score = 0.7
        else:  # High volatility
            vol_score = 0.4

        strength += vol_score * 0.1

        return min(1.0, strength)

    def _calculate_levels(
        self,
        signal_type: SignalType,
        market_data: MarketData,
        sentiment: 'CommoditySentiment'
    ) -> tuple[float, float, float]:
        """
        Calculate entry, stop-loss, and take-profit levels.

        Returns: (entry, stop, target)
        """
        price = market_data.current_price
        atr = market_data.atr_20

        if signal_type == SignalType.LONG:
            # Long position
            entry = price  # Enter at current price (or can set limit order)

            # Stop-loss: 2× ATR below entry
            stop = entry - (2 * atr)

            # Target: Risk/reward ratio (default 2:1)
            risk = entry - stop
            target = entry + (risk * self.default_risk_reward)

        else:  # SHORT
            # Short position
            entry = price

            # Stop-loss: 2× ATR above entry
            stop = entry + (2 * atr)

            # Target: Risk/reward ratio
            risk = stop - entry
            target = entry - (risk * self.default_risk_reward)

        return entry, stop, target

    def _build_rationale(
        self,
        sentiment: 'CommoditySentiment',
        market_data: MarketData,
        signal_type: SignalType
    ) -> str:
        """
        Build human-readable rationale for signal.
        """
        direction = "LONG" if signal_type == SignalType.LONG else "SHORT"

        rationale = f"{direction} signal based on:\n"
        rationale += f"- Sentiment: {sentiment.overall_direction.value} (confidence: {sentiment.confidence:.0%})\n"
        rationale += f"- Price sentiment: {sentiment.price_sentiment.reasoning}\n"

        # Add key factors
        key_factors = sentiment.key_factors.get(
            'bullish' if signal_type == SignalType.LONG else 'bearish',
            []
        )
        if key_factors:
            rationale += f"- Key factors: {', '.join(key_factors[:3])}\n"

        # Add price confirmation
        price = market_data.current_price
        sma50 = market_data.sma_50

        if signal_type == SignalType.LONG:
            rationale += f"- Price confirmation: ${price:.2f} above SMA50 ${sma50:.2f}\n"
        else:
            rationale += f"- Price confirmation: ${price:.2f} below SMA50 ${sma50:.2f}\n"

        return rationale

    def _determine_time_horizon(
        self,
        sentiment: 'CommoditySentiment'
    ) -> TimeHorizon:
        """
        Determine appropriate time horizon from sentiment.
        """
        if sentiment.time_horizon == sentiment.time_horizon.SHORT_TERM:
            return TimeHorizon.SWING
        elif sentiment.time_horizon == sentiment.time_horizon.MEDIUM_TERM:
            return TimeHorizon.POSITION
        else:
            return TimeHorizon.POSITION
```

</div>
</div>

### Signal Validation and Backtesting


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">signalvalidator.py</span>
</div>

```python
class SignalValidator:
    """
    Validate signals against historical data.
    """

    def __init__(self):
        self.signal_history = []
        self.performance_metrics = {}

    def backtest_signal(
        self,
        signal: TradingSignal,
        historical_prices: pd.DataFrame,
        holding_period_days: int = 5
    ) -> Dict[str, any]:
        """
        Backtest signal against historical price data.

        Args:
            signal: Generated trading signal
            historical_prices: DataFrame with OHLCV data
            holding_period_days: How long to hold position

        Returns:
            Performance metrics for this signal
        """
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit

        # Simulate trade over holding period
        entry_date = signal.generated_at
        exit_date = entry_date + timedelta(days=holding_period_days)

        # Get price data for holding period
        period_data = historical_prices[
            (historical_prices.index >= entry_date) &
            (historical_prices.index <= exit_date)
        ]

        if len(period_data) == 0:
            return {'status': 'no_data'}

        # Check if stop or target hit
        stopped_out = False
        target_hit = False
        exit_price = None

        for date, row in period_data.iterrows():
            if signal.signal_type == SignalType.LONG:
                # Long position
                if row['Low'] <= stop_loss:
                    stopped_out = True
                    exit_price = stop_loss
                    break
                if row['High'] >= take_profit:
                    target_hit = True
                    exit_price = take_profit
                    break
            else:  # SHORT
                # Short position
                if row['High'] >= stop_loss:
                    stopped_out = True
                    exit_price = stop_loss
                    break
                if row['Low'] <= take_profit:
                    target_hit = True
                    exit_price = take_profit
                    break

        # If neither hit, exit at end of period
        if not stopped_out and not target_hit:
            exit_price = period_data.iloc[-1]['Close']

        # Calculate P&L
        if signal.signal_type == SignalType.LONG:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        return {
            'status': 'completed',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'stopped_out': stopped_out,
            'target_hit': target_hit,
            'holding_days': len(period_data),
            'max_favorable_excursion': self._calculate_mfe(period_data, signal),
            'max_adverse_excursion': self._calculate_mae(period_data, signal)
        }

    def _calculate_mfe(
        self,
        period_data: pd.DataFrame,
        signal: TradingSignal
    ) -> float:
        """Maximum Favorable Excursion - best price achieved."""
        if signal.signal_type == SignalType.LONG:
            best_price = period_data['High'].max()
            return (best_price - signal.entry_price) / signal.entry_price
        else:
            best_price = period_data['Low'].min()
            return (signal.entry_price - best_price) / signal.entry_price

    def _calculate_mae(
        self,
        period_data: pd.DataFrame,
        signal: TradingSignal
    ) -> float:
        """Maximum Adverse Excursion - worst price achieved."""
        if signal.signal_type == SignalType.LONG:
            worst_price = period_data['Low'].min()
            return (worst_price - signal.entry_price) / signal.entry_price
        else:
            worst_price = period_data['High'].max()
            return (signal.entry_price - worst_price) / signal.entry_price
```

</div>
</div>

### Signal Portfolio Management


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">signalportfolio.py</span>
</div>

```python
class SignalPortfolio:
    """
    Manage multiple signals across commodities.
    """

    def __init__(self, total_capital: float, max_positions: int = 5):
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.active_signals = []
        self.closed_signals = []

    def add_signal(self, signal: TradingSignal, commodity: str) -> bool:
        """
        Add signal to portfolio if it passes risk checks.

        Returns: True if added, False if rejected
        """
        # Check if already at max positions
        if len(self.active_signals) >= self.max_positions:
            return False

        # Calculate position size
        position_size = self._calculate_position_size(signal)

        # Check if position size is acceptable
        if position_size == 0:
            return False

        # Add to active signals
        self.active_signals.append({
            'signal': signal,
            'commodity': commodity,
            'position_size': position_size,
            'entry_date': datetime.now()
        })

        return True

    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size based on Kelly Criterion / fixed fractional.

        Position size = (Capital × Signal Strength) / Risk Per Trade
        """
        # Risk per trade (1-2% of capital)
        risk_per_trade = self.total_capital * 0.02

        # Calculate shares/contracts
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)

        if risk_per_unit == 0:
            return 0

        # Base position size from risk management
        base_position = risk_per_trade / risk_per_unit

        # Adjust by signal strength
        adjusted_position = base_position * signal.strength

        # Cap at maximum position size (10% of capital)
        max_position_value = self.total_capital * 0.10
        max_units = max_position_value / signal.entry_price

        final_position = min(adjusted_position, max_units)

        return final_position

    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get current portfolio status."""
        total_exposure = sum(
            pos['position_size'] * pos['signal'].entry_price
            for pos in self.active_signals
        )

        return {
            'active_positions': len(self.active_signals),
            'max_positions': self.max_positions,
            'total_capital': self.total_capital,
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / self.total_capital,
            'available_capital': self.total_capital - total_exposure
        }
```

</div>

## Common Pitfalls

**1. Acting on Low-Confidence Sentiment**
- **Problem**: Trading every sentiment signal regardless of confidence
- **Why it happens**: Wanting to "stay active" in markets
- **Solution**: Set minimum confidence threshold (0.7+); quality over quantity

**2. Ignoring Price Confirmation**
- **Problem**: Going long on bullish sentiment while price is crashing
- **Why it happens**: Trusting sentiment alone without checking price action
- **Solution**: Require price confirmation (trend alignment, breakouts)

**3. No Stop-Loss Discipline**
- **Problem**: Not setting or honoring stop-loss levels
- **Why it happens**: Hope that losing trades will recover
- **Solution**: Calculate stops before entry; use automated stop orders

**4. Oversizing Positions**
- **Problem**: Risking too much capital on single signal
- **Why it happens**: Overconfidence in signal accuracy
- **Solution**: Fixed fractional position sizing (1-2% risk per trade)

**5. Ignoring Time Decay**
- **Problem**: Holding positions based on old sentiment
- **Why it happens**: Not updating signals as new information arrives
- **Solution**: Set signal expiry times; re-evaluate with fresh sentiment

## Connections

**Builds on:**
- Sentiment extraction (input to signal construction)
- Risk management principles (position sizing, stop-loss)
- Technical analysis (price confirmation, support/resistance)

**Leads to:**
- Portfolio optimization (combining multiple signals)
- Execution strategies (optimal entry/exit timing)
- Performance attribution (understanding signal profitability)

**Related to:**
- Quantitative trading (systematic signal generation)
- Options strategies (alternative ways to express signals)
- Market microstructure (execution quality)

## Practice Problems

1. **Signal Strength Calculation**
   - Given sentiment with confidence 0.85 and 4 bullish factors
   - Current RSI: 55, Price above SMA50
   - ATR/Price: 3% (moderate volatility)
   - Calculate signal strength using the weighted formula

2. **Entry/Exit Level Design**
   - Crude oil trading at $85.00
   - ATR(20) = $2.50
   - Bullish signal from sentiment
   - Calculate: Entry, Stop-Loss (2× ATR), Take-Profit (2:1 R/R)

3. **Position Sizing**
   - Portfolio capital: $100,000
   - Risk per trade: 2% ($2,000)
   - Signal: Long crude oil at $85, stop at $82
   - Calculate position size in contracts (1 contract = 1000 barrels)

4. **Signal Validation**
   - Backtest this signal over 20 historical instances:
     - 12 winners (avg +3.5%)
     - 8 losers (avg -1.8%)
   - Calculate: Win rate, Avg win, Avg loss, Expectancy
   - Is this a profitable signal?

5. **Portfolio Risk Management**
   - You have 3 active signals:
     - Long crude oil: $30k exposure
     - Short natural gas: $25k exposure
     - Long corn: $20k exposure
   - Total capital: $100k
   - Is this acceptable diversification? Why or why not?

<div class="callout-insight">

**Insight:** Understanding building trading signals from commodity sentiment is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Further Reading

**Trading Signal Design:**
- "Quantitative Trading" (Ernie Chan) - Systematic signal construction
- "Evidence-Based Technical Analysis" (David Aronson) - Validating signals

**Risk Management:**
- "The Mathematics of Money Management" (Ralph Vince) - Position sizing
- "Trade Your Way to Financial Freedom" (Van Tharp) - Risk/reward management

**Backtesting:**
- "Advances in Financial Machine Learning" (Marcos López de Prado) - Avoiding overfitting
- "Quantitative Trading Strategies" - Statistical validation

**Sentiment-Based Trading:**
- "Trading on Sentiment" (Richard Peterson) - Behavioral finance applications
- "Social Media Sentiment for Commodity Markets" (2024 research)

**Production Systems:**
- "Building Winning Algorithmic Trading Systems" (Kevin Davey) - Systematic trading
- "Algorithmic Trading and DMA" (Barry Johnson) - Execution strategies

---

## Conceptual Practice Questions

1. How do you aggregate article-level sentiments into a tradeable market signal?

2. What decay function would you use for time-weighting sentiment scores and why?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./03_signal_construction_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_news_sentiment.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_news_processing.md">
  <div class="link-card-title">01 News Processing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_news_sentiment.md">
  <div class="link-card-title">01 News Sentiment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

