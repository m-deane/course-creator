# Backtesting LLM-Generated Signals

> **Reading time:** ~12 min | **Module:** Module 5: Signals | **Prerequisites:** Modules 0-4

<div class="callout-key">

**Key Concept Summary:** Backtesting validates whether LLM-generated trading signals would have been profitable historically, accounting for transaction costs, slippage, and realistic execution constraints. Unlike traditional backtesting, LLM backtests require special handling of information leakage, prompt stability, an...

</div>

## In Brief

Backtesting validates whether LLM-generated trading signals would have been profitable historically, accounting for transaction costs, slippage, and realistic execution constraints. Unlike traditional backtesting, LLM backtests require special handling of information leakage, prompt stability, and computational cost since LLM calls cannot be cheaply replicated millions of times.

<div class="callout-insight">

**Insight:** LLM backtesting faces unique challenges: (1) LLM outputs are non-deterministic even with the same prompt, (2) computational cost limits exhaustive testing, (3) the LLM's training data may include information from your backtest period (data leakage), (4) prompts must remain stable across years despite changing market conditions. The solution: cache LLM responses, use walk-forward validation, detect temporal data leakage, and measure prompt robustness.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

## Formal Definition

### Backtest Framework

**Signal Generation Process:**
$$s_t = \text{LLM}(C_t, \theta)$$

Where:
- $s_t$: Signal at time t (direction, strength, risk params)
- $C_t$: Context available at time t (must enforce causality)
- $\theta$: LLM parameters (model, temperature, prompt)

**P&L Calculation:**

For a long signal at time t:
$$\text{PnL}_t = n_t \times (p_{t+1} - p_t) - \text{TC}_t - \text{Slip}_t$$

Where:
- $n_t$: Position size (contracts/shares)
- $p_t$: Price at time t
- $\text{TC}_t$: Transaction costs (commissions, fees)
- $\text{Slip}_t$: Slippage (difference between theoretical and actual execution)

**Performance Metrics:**

1. **Sharpe Ratio:** $\text{SR} = \frac{\mu_r}{\sigma_r} \sqrt{252}$ (annualized)

2. **Maximum Drawdown:** $\text{MDD} = \max_{t} \left( \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t} \right)$

3. **Win Rate:** $\text{WR} = \frac{\text{Winning Trades}}{\text{Total Trades}}$

4. **Profit Factor:** $\text{PF} = \frac{\sum \text{Wins}}{\sum |\text{Losses}|}$

5. **Calmar Ratio:** $\text{CR} = \frac{\text{Annual Return}}{\text{MDD}}$

### Walk-Forward Validation

To prevent overfitting:

<div class="flow">
<div class="flow-step mint">1. Training Window:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Validation Window:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Test Window:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Roll Forward:</div>
</div>


1. **Training Window:** Fit calibration, tune prompt (t - N to t - M)
2. **Validation Window:** Select best prompt variant (t - M to t - K)
3. **Test Window:** Generate signals, measure performance (t - K to t)
4. **Roll Forward:** Shift windows, repeat

**No Look-Ahead Bias:**
$$C_t = \{x_s : s \leq t\}$$

Only information available at time t can be used in prompt.

### Computational Efficiency

**Cost per backtest:**
$$\text{Cost} = n_{\text{signals}} \times c_{\text{LLM}} \times n_{\text{variants}}$$

For 1000 signals, $0.01 per call, 10 variants → $100/backtest

**Solution: Response Caching**
- Hash (prompt, context, timestamp) → cache response
- Reuse cached responses across backtest iterations
- Reduces cost by 95% for parameter sweeps

### Traditional vs LLM Backtesting

**Traditional (Fast):**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Compute indicator once
indicator = compute_RSI(prices)
# Test thousands of parameter combinations
for threshold in range(20, 80):
    signals = indicator > threshold
    pnl = backtest(signals, prices)
```

</div>
Time: Seconds for thousands of tests

**LLM-Based (Expensive):**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Must call LLM for each signal generation
for date in dates:
    context = get_context(date)
    signal = LLM(context)  # $$$
    pnl = execute(signal, prices)
```

</div>
Time: Hours for single test

**Hybrid Approach:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Cache LLM responses
cache = {}
for date in dates:
    context = get_context(date)
    key = hash(context, prompt)

    if key not in cache:
        cache[key] = LLM(context)

    signal = cache[key]
    pnl = execute(signal, prices)
```

</div>
Time: Minutes, reusable across parameter tests

### Walk-Forward Prevents Overfitting

**Bad: In-Sample Overfitting**
```
[=============== Training (5 years) ================]
Test hundreds of prompt variants, pick best
Report performance on same 5 years → OVERFITTED
```

**Good: Walk-Forward**
```
Year 1-3: Training  →  Year 4: Validation  →  Year 5: Test
Year 2-4: Training  →  Year 5: Validation  →  Year 6: Test
...
Report: Average of out-of-sample test periods
```

## Code Implementation

### Backtest Engine with Caching


<span class="filename">from.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from pathlib import Path
from anthropic import Anthropic

@dataclass
class Trade:
    """Single trade record."""
    entry_date: datetime
    exit_date: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    signal_strength: float
    reason: str

@dataclass
class BacktestResult:
    """Backtest performance summary."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    trades: List[Trade]
    equity_curve: pd.Series


class LLMSignalCache:
    """
    Cache LLM responses to reduce API costs during backtesting.
    """

    def __init__(self, cache_dir: str = './llm_backtest_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _hash_key(self, prompt: str, context: str, model: str) -> str:
        """Generate cache key from inputs."""
        content = f"{model}|{prompt}|{context}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, context: str, model: str) -> Optional[Dict]:
        """Retrieve cached response."""
        key = self._hash_key(prompt, context, model)

        # Check in-memory cache
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                response = pickle.load(f)
                self.cache[key] = response
                self.hits += 1
                return response

        self.misses += 1
        return None

    def set(self, prompt: str, context: str, model: str, response: Dict):
        """Cache response."""
        key = self._hash_key(prompt, context, model)

        # Store in memory
        self.cache[key] = response

        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)

    def stats(self) -> Dict:
        """Cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class LLMBacktester:
    """
    Backtest LLM-generated trading signals.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.0005,  # 5 bps
        slippage_bps: float = 2.0
    ):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.cache = LLMSignalCache()

    def generate_signal(
        self,
        date: datetime,
        context: str,
        prompt_template: str,
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict:
        """
        Generate signal with caching.

        Args:
            date: Signal date
            context: Market context at this date
            prompt_template: Prompt template
            model: LLM model

        Returns:
            Parsed signal dictionary
        """
        # Build full prompt
        full_prompt = prompt_template.format(
            date=date.strftime('%Y-%m-%d'),
            context=context
        )

        # Check cache
        cached = self.cache.get(full_prompt, context, model)
        if cached is not None:
            return cached

        # Call LLM
        response = self.client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": full_prompt}]
        )

        # Parse response
        try:
            signal = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            # Fallback parsing
            text = response.content[0].text.lower()
            signal = {
                'direction': 'neutral',
                'strength': 0.0,
                'reasoning': text
            }

        # Cache response
        self.cache.set(full_prompt, context, model, signal)

        return signal

    def execute_trade(
        self,
        signal: Dict,
        entry_price: float,
        exit_price: float,
        holding_period: int,
        capital: float
    ) -> Trade:
        """
        Execute trade based on signal.

        Args:
            signal: Signal dictionary
            entry_price: Entry price
            exit_price: Exit price
            holding_period: Days held
            capital: Available capital

        Returns:
            Trade object with P&L
        """
        direction = signal.get('direction', 'neutral')
        strength = signal.get('strength', 0.5)

        if direction == 'neutral':
            # No trade
            return None

        # Position sizing
        position_pct = 0.10 * strength  # Max 10% of capital
        position_value = capital * position_pct
        size = position_value / entry_price

        # Transaction costs
        entry_commission = position_value * self.commission_rate
        exit_commission = position_value * self.commission_rate
        total_commission = entry_commission + exit_commission

        # Slippage
        entry_slip = entry_price * (self.slippage_bps / 10000)
        exit_slip = exit_price * (self.slippage_bps / 10000)

        # P&L calculation
        if direction == 'long':
            actual_entry = entry_price + entry_slip
            actual_exit = exit_price - exit_slip
            gross_pnl = size * (actual_exit - actual_entry)
        else:  # short
            actual_entry = entry_price - entry_slip
            actual_exit = exit_price + exit_slip
            gross_pnl = size * (actual_entry - actual_exit)

        net_pnl = gross_pnl - total_commission

        return Trade(
            entry_date=datetime.now(),  # Would be actual dates
            exit_date=datetime.now() + timedelta(days=holding_period),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=net_pnl,
            signal_strength=strength,
            reason=signal.get('reasoning', '')
        )

    def run_backtest(
        self,
        data: pd.DataFrame,
        prompt_template: str,
        signal_column: str = 'context',
        price_column: str = 'close',
        holding_period: int = 5
    ) -> BacktestResult:
        """
        Run full backtest.

        Args:
            data: DataFrame with price data and context
            prompt_template: Prompt template for signal generation
            signal_column: Column containing context for LLM
            price_column: Price column for execution
            holding_period: Days to hold position

        Returns:
            BacktestResult object
        """
        trades = []
        equity = self.initial_capital
        equity_curve = []

        for i in range(len(data) - holding_period):
            row = data.iloc[i]
            future_row = data.iloc[i + holding_period]

            # Generate signal
            signal = self.generate_signal(
                date=row.name if isinstance(row.name, datetime) else datetime.now(),
                context=row[signal_column],
                prompt_template=prompt_template
            )

            # Execute trade
            trade = self.execute_trade(
                signal=signal,
                entry_price=row[price_column],
                exit_price=future_row[price_column],
                holding_period=holding_period,
                capital=equity
            )

            if trade:
                equity += trade.pnl
                trades.append(trade)

            equity_curve.append(equity)

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> BacktestResult:
        """Calculate performance metrics."""
        if not trades:
            return BacktestResult(
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                num_trades=0,
                trades=[],
                equity_curve=pd.Series()
            )

        # Total return
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Returns series
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        # Sharpe ratio
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Maximum drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades)

        # Profit factor
        total_wins = sum(t.pnl for t in trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity_series
        )

    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        prompt_template: str,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21
    ) -> List[BacktestResult]:
        """
        Walk-forward validation to prevent overfitting.

        Args:
            data: Full dataset
            prompt_template: Prompt template
            train_days: Training window size
            test_days: Test window size
            step_days: Days to step forward each iteration

        Returns:
            List of test period results
        """
        results = []
        start_idx = train_days

        while start_idx + test_days < len(data):
            # Training data (for calibration, not used here but could be)
            train_data = data.iloc[start_idx - train_days:start_idx]

            # Test data
            test_data = data.iloc[start_idx:start_idx + test_days]

            # Run backtest on test period
            result = self.run_backtest(test_data, prompt_template)
            results.append(result)

            # Step forward
            start_idx += step_days

        return results


# Example Usage

# Generate synthetic price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
n = len(dates)

prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
data = pd.DataFrame({
    'close': prices,
    'context': [f"Price trend analysis for {date.strftime('%Y-%m-%d')}" for date in dates]
}, index=dates)

# Prompt template
prompt_template = """
Analyze crude oil market for date: {date}

Context: {context}

Recent price action suggests volatility. Generate trading signal as JSON:
{{
  "direction": "long" | "short" | "neutral",
  "strength": 0.0-1.0,
  "reasoning": "Brief explanation"
}}
"""

# Run backtest
print("=" * 70)
print("LLM SIGNAL BACKTESTING")
print("=" * 70)

backtester = LLMBacktester(
    anthropic_api_key="your-key-here",
    initial_capital=1_000_000
)

result = backtester.run_backtest(
    data=data.iloc[:100],  # First 100 days for demo
    prompt_template=prompt_template,
    holding_period=5
)

print(f"\nBacktest Results:")
print(f"  Total Return: {result.total_return*100:.2f}%")
print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")
print(f"  Win Rate: {result.win_rate*100:.1f}%")
print(f"  Profit Factor: {result.profit_factor:.2f}")
print(f"  Number of Trades: {result.num_trades}")

# Cache statistics
cache_stats = backtester.cache.stats()
print(f"\nCache Statistics:")
print(f"  Hits: {cache_stats['hits']}")
print(f"  Misses: {cache_stats['misses']}")
print(f"  Hit Rate: {cache_stats['hit_rate']*100:.1f}%")

# Walk-forward validation
print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION")
print("=" * 70)

wf_results = backtester.walk_forward_validation(
    data=data.iloc[:500],
    prompt_template=prompt_template,
    train_days=100,
    test_days=30,
    step_days=30
)

print(f"\nNumber of test periods: {len(wf_results)}")
print(f"Average Sharpe Ratio: {np.mean([r.sharpe_ratio for r in wf_results]):.2f}")
print(f"Average Win Rate: {np.mean([r.win_rate for r in wf_results])*100:.1f}%")
print(f"Worst Drawdown: {min([r.max_drawdown for r in wf_results])*100:.2f}%")
```

</div>

## Common Pitfalls

**1. Temporal Data Leakage**
- Problem: LLM trained on data from your backtest period
- Symptom: Suspiciously good performance that doesn't translate to live trading
- Solution: Test on recent data post-LLM training cutoff, use older LLM versions for historical tests

**2. Look-Ahead Bias**
- Problem: Context includes information not available at signal generation time
- Symptom: Backtest shows 80% win rate, live trading shows 45%
- Solution: Strict timestamp checks, only use data published before signal date

**3. Survivor Bias**
- Problem: Testing only on commodities that still exist/trade actively
- Symptom: Overestimating strategy robustness
- Solution: Include delisted contracts, test across multiple commodities

**4. Ignoring Transaction Costs**
- Problem: Not accounting for slippage, commissions, market impact
- Symptom: Backtest profitable, live trading marginal after costs
- Solution: Conservative estimates (5-10 bps slippage, realistic commissions)

**5. Prompt Instability**
- Problem: Slight prompt changes cause wildly different signals
- Symptom: Cannot reproduce backtest results, sensitive to wording
- Solution: Test prompt robustness (paraphrased versions), use temperature=0 for determinism

## Connections

**Builds on:**
- Module 5.1: Signal frameworks (signals being backtested)
- Module 5.2: Confidence scoring (position sizing in backtest)
- Quantitative finance (Sharpe ratio, drawdown, performance measurement)

**Leads to:**
- Module 6.1: Production deployment (validated strategies go live)
- Module 6.2: Monitoring (track live vs backtest performance divergence)
- Portfolio construction (combine multiple backtested strategies)

**Related concepts:**
- Walk-forward analysis (time series cross-validation)
- Monte Carlo simulation (randomized backtest scenarios)
- Regime detection (adaptive backtesting across market states)

## Practice Problems

1. **Cost Analysis**
   Backtest requires 1000 LLM calls at $0.01 each.
   Cache hit rate: 60% after parameter sweep.
   How many parameter variants can you test for $500 budget?

2. **Walk-Forward Design**
   4 years of data (1000 trading days).
   Train: 1 year, Test: 3 months, Step: 1 month.
   How many test periods? What % of data is tested out-of-sample?

3. **Drawdown Recovery**
   Strategy hits 30% drawdown.
   What return needed to recover to breakeven?
   If Sharpe=1.5, expected time to recover?

4. **Signal Timing**
   LLM signal generated at market close using news available by 4pm.
   Can you execute at today's close or must wait until tomorrow's open?
   Impact on backtest if you assume same-day execution?

5. **Cache Invalidation**
   Prompt includes "recent trend" which depends on last 10 days.
   When is cached response invalid?
   Design cache key to handle this?

<div class="callout-insight">

**Insight:** Understanding backtesting llm-generated signals is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Backtesting Fundamentals:**
1. **"Evidence-Based Technical Analysis" by David Aronson** - Rigorous backtest methodology
2. **"Advances in Financial Machine Learning" by Marcos López de Prado** - Backtesting pitfalls
3. **"Quantitative Trading" by Ernest Chan** - Walk-forward validation

**Transaction Cost Analysis:**
4. **"Algorithmic Trading and DMA" by Barry Johnson** - Market microstructure, slippage
5. **"Trading and Exchanges" by Larry Harris** - Transaction costs

**LLM-Specific:**
6. **"Evaluating Large Language Models Trained on Code"** - Evaluation methodology
7. **"Beyond Accuracy: Behavioral Testing of NLP Models"** - Robustness testing
8. **"Time Series Cross-Validation for LLMs"** - Temporal validation

**Statistical Testing:**
9. **"Testing Trading Strategies" by Tim Bauer** - Statistical significance
10. **"The Deflated Sharpe Ratio" by Bailey & López de Prado** - Multiple testing correction

**Risk Management:**
11. **"Risk and Portfolio Management with Econometrics" by Roncalli** - Risk metrics
12. **"The Mathematics of Money Management" by Ralph Vince** - Position sizing in backtests

*"Backtest like you'll trade. Trade like you backtested."*

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_backtesting_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_signal_generation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_signal_frameworks.md">
  <div class="link-card-title">01 Signal Frameworks</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_signal_generation.md">
  <div class="link-card-title">01 Signal Generation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

