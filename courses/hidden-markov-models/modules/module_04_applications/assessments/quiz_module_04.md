# Quiz: Module 4 - Financial Applications

**Course:** Hidden Markov Models
**Module:** 4 - Regime Detection and Trading Applications
**Total Points:** 100
**Estimated Time:** 30 minutes
**Attempts Allowed:** 2

---

## Instructions

This quiz assesses your understanding of HMM applications in finance, including regime detection, volatility modeling, trading strategies, and portfolio allocation. Questions focus on practical implementation and interpretation.

---

## Section 1: Market Regime Detection (25 points)

### Question 1 (10 points)

A hedge fund fits a 3-state HMM to S&P 500 returns and identifies the following regimes:

```
State 1: μ = 0.08%, σ = 0.9%, π^∞ = 0.55
State 2: μ = 0.01%, σ = 0.7%, π^∞ = 0.30
State 3: μ = -0.15%, σ = 2.5%, π^∞ = 0.15

Transition matrix A:
       S1    S2    S3
S1  [0.96  0.03  0.01]
S2  [0.10  0.85  0.05]
S3  [0.10  0.15  0.75]
```

**Part A (5 points):** Label and interpret each state in financial terms.

**Answer:**

**State 1 - Bull Market:**
- Positive mean return (0.08% daily ≈ 20% annualized)
- Low-moderate volatility (0.9% ≈ 14.3% annualized)
- Most persistent (96% stay probability)
- Most common regime (55% of time)
- Typical expansion/bull market conditions

**State 2 - Neutral/Consolidation:**
- Near-zero returns (0.01% ≈ 2.5% annualized)
- Lowest volatility (0.7% ≈ 11.1% annualized)
- Moderate persistence (85%)
- Second most common (30% of time)
- Sideways/ranging market, low conviction

**State 3 - Bear/Crisis:**
- Negative returns (-0.15% ≈ -37% annualized)
- High volatility (2.5% ≈ 39.7% annualized)
- Least persistent (75%)
- Least common (15% of time)
- Crash/crisis conditions, resolves faster than bull markets

**Part B (5 points):** Based on the transition matrix, which regime transition is most likely when exiting State 3 (Bear)?

**Answer: State 1 (Bull) - 10% probability**

**Explanation:**
From State 3, the exit probabilities are:
- To State 1: 0.10 (10%)
- To State 2: 0.15 (15%)
- Stay in State 3: 0.75 (75%)

When conditioned on exiting (not staying), the probabilities are:
- P(S1 | exit S3) = 0.10 / (0.10 + 0.15) = 0.10 / 0.25 = 0.40 (40%)
- P(S2 | exit S3) = 0.15 / (0.10 + 0.15) = 0.15 / 0.25 = 0.60 (60%)

Actually, **State 2 (Neutral) is most likely** when exiting State 3.

This makes financial sense:
- Markets rarely bounce directly from crisis to bull
- Transition through neutral/consolidation phase
- Builds base before resuming uptrend
- Consistent with historical market recoveries

---

### Question 2 (8 points)

Why is using the **smoothing distribution** P(S_t | O_{1:T}, λ) preferred over the **filtering distribution** P(S_t | O_{1:t}, λ) for historical regime analysis?

A) Smoothing is faster to compute
B) Smoothing uses all available data, including future observations
C) Filtering requires the backward algorithm
D) Smoothing gives higher probabilities

**Answer: B**

**Explanation:**

**Filtering: P(S_t | O_{1:t}, λ)**
- Uses only observations up to time t
- "Real-time" inference - what did we know at time t?
- Computed using forward algorithm: γ_t^{filt}(i) ∝ α_t(i)

**Smoothing: P(S_t | O_{1:T}, λ)**
- Uses ALL observations (past, present, future)
- "Retrospective" analysis - what do we know looking back?
- Computed using forward-backward: γ_t(i) ∝ α_t(i) × β_t(i)

**Why smoothing is better for historical analysis:**

1. **More accurate:** Future observations provide additional information
   - If t=50 had ambiguous regime, observations at t=51-100 help clarify

2. **Less noisy:** Smoothing reduces uncertainty
   - Uses data on both sides of time t
   - More stable regime assignments

3. **Better for research:** Understanding historical regimes benefits from hindsight
   - "Was 2008 actually a crisis?" - use all data from 2008-2009
   - Academic studies should use best available inference

**When to use filtering:**

1. **Live trading:** Only past data available in real-time
2. **Backtesting:** Simulate realistic trading with causal information only
3. **Online learning:** Sequential decision-making

**Example:**
On Oct 15, 2008 (mid-crisis):
- Filtering P(S_t=Crisis | past returns) ≈ 0.65 (uncertain, just starting)
- Smoothing P(S_t=Crisis | all 2008 returns) ≈ 0.98 (clear crisis in retrospect)

---

### Question 3 (7 points)

A portfolio manager uses Viterbi decoding to identify historical regimes. On a particular day, the smoothing probabilities are:

```
P(S_t = Bull | O_{1:T}) = 0.55
P(S_t = Neutral | O_{1:T}) = 0.40
P(S_t = Bear | O_{1:T}) = 0.05
```

The Viterbi algorithm assigns this day to the Neutral state. How is this possible?

**Answer:**

The Viterbi algorithm finds the **globally most likely state sequence**, not the most likely state at each individual time point.

**Scenario:**

The Viterbi path might be:
- t-1: Neutral (certain)
- t: **Neutral** (Viterbi choice)
- t+1: Neutral (certain)

While the marginal probabilities suggest Bull (55%) is most likely at time t, the globally optimal path is Neutral-Neutral-Neutral because:

**Transition constraints:**
- P(Neutral → Neutral) might be very high (e.g., 0.85)
- P(Neutral → Bull) might be very low (e.g., 0.10)
- P(Bull → Neutral) might be very low (e.g., 0.03)

**Joint probability comparison:**

Path 1: Neutral → Bull → Neutral
- P ∝ 0.10 × 0.55 × 0.03 = 0.00165

Path 2: Neutral → Neutral → Neutral
- P ∝ 0.85 × 0.40 × 0.85 = 0.289

Path 2 is much more probable (289× higher) despite lower marginal at time t.

**Key insight:**
- **Marginal decoding:** max_i P(S_t = i | O) at each t independently
- **Viterbi decoding:** max_{S_{1:T}} P(S_{1:T} | O) globally
- Viterbi enforces transition dynamics; marginals ignore them

**Financial interpretation:**
- Regimes have persistence (momentum)
- A single day's return might suggest Bull, but context (before/after) indicates Neutral
- Viterbi gives more coherent regime narratives

---

## Section 2: Volatility State Modeling (25 points)

### Question 4 (10 points)

A risk manager models VIX dynamics using a 2-state HMM:

```
State 1 (Low Vol): μ_1 = 12, σ_1 = 2
State 2 (High Vol): μ_2 = 28, σ_2 = 6

A = [0.97  0.03]
    [0.08  0.92]
```

Current VIX observations: [14, 15, 16, 18, 22, 25, 29]

**Part A (5 points):** Without full calculation, when would you expect the Viterbi algorithm to indicate a regime switch from Low Vol to High Vol?

**Answer:**

The switch likely occurs between t=5 and t=6 (22 → 25) or between t=4 and t=5 (18 → 22).

**Reasoning:**

**Observations 1-4 (14, 15, 16, 18):**
- All close to Low Vol mean (12)
- Within 1-2 standard deviations of Low Vol
- Far from High Vol mean (28)
- Clear Low Vol state

**Observation 5 (22):**
- Ambiguous: between the two means
- z_low = (22-12)/2 = 5 (extremely unlikely for Low Vol)
- z_high = (22-28)/6 = -1 (plausible for High Vol)
- Evidence mounting for High Vol

**Observations 6-7 (25, 29):**
- Close to High Vol mean (28)
- Far from Low Vol mean (12)
- Clear High Vol state

**Transition dynamics:**
- Low self-transition (0.97) means switches are rare
- Once evidence overwhelms this prior, switch occurs
- Viterbi finds the single switch point that maximizes total path probability
- Most likely at t=5 or t=6 when VIX crosses ~20 (midpoint)

**Part B (5 points):** If the manager wants to implement a volatility-targeting strategy that adjusts portfolio leverage based on the detected regime, should they use filtering or smoothing?

**Answer: Filtering P(S_t | O_{1:t}, λ)**

**Explanation:**

**For live trading, use filtering:**
- Only past and current data available
- Real-time decision-making
- Causal inference (no look-ahead bias)

**Implementation:**
```
At time t:
1. Observe VIX_t
2. Compute α_t(i) using forward algorithm
3. Compute filtering: P(S_t=High | O_{1:t}) = α_t(High) / Σ_i α_t(i)
4. If P(S_t=High | O_{1:t}) > threshold (e.g., 0.6):
   - Reduce leverage (e.g., from 1.5x to 1.0x)
   - Shift to defensive positions
5. If P(S_t=Low | O_{1:t}) > threshold:
   - Increase leverage (e.g., from 1.0x to 1.5x)
   - More aggressive positioning
```

**Why not smoothing:**
- Requires future observations (unavailable in real-time)
- Creates look-ahead bias if used in backtesting
- Only useful for historical analysis, not trading

**Practical considerations:**
- Add transaction costs to avoid excessive trading
- Use regime probability threshold (e.g., P > 0.7) to reduce false signals
- Combine with other risk metrics (e.g., position limits)

---

### Question 5 (8 points)

Explain how a 2-state volatility HMM differs from a GARCH model for volatility forecasting.

**Answer:**

**Hidden Markov Model (HMM):**

**Structure:**
- Discrete volatility regimes (e.g., Low Vol, High Vol)
- Markov switching between regimes
- Volatility jumps between distinct levels

**Dynamics:**
- Volatility changes via regime switches (discrete jumps)
- Within-regime: volatility is relatively constant
- Captures structural breaks and regime persistence

**Strengths:**
- Natural for regime-based strategies
- Interpretable states (bull/bear, calm/crisis)
- Captures mean-reverting regimes

**Weaknesses:**
- Discrete states may oversimplify continuous volatility dynamics
- Sudden jumps may not match gradual volatility changes
- Requires choosing number of states

**GARCH Model (Generalized Autoregressive Conditional Heteroskedasticity):**

**Structure:**
- σ_t² = ω + α × r_{t-1}² + β × σ_{t-1}²
- Continuous volatility evolution
- Volatility depends on past shocks and past volatility

**Dynamics:**
- Volatility changes gradually and continuously
- Volatility clustering: high volatility persists
- Mean-reverting to long-run average

**Strengths:**
- Smooth volatility dynamics
- Well-established econometric framework
- Captures volatility clustering naturally

**Weaknesses:**
- No distinct regimes (harder to interpret)
- May not capture structural breaks
- Assumes single data-generating process

**Comparison:**

| Aspect | HMM | GARCH |
|--------|-----|-------|
| Volatility | Discrete regimes | Continuous |
| Transitions | Sudden jumps | Gradual evolution |
| Interpretation | High/Low states | Numerical forecast |
| Regime shifts | Explicit | Implicit |
| Parameters | Regime-specific | Global |

**Hybrid approach: Markov-Switching GARCH**
- GARCH dynamics within each HMM regime
- Combines benefits: regime interpretation + continuous dynamics
- More complex but more realistic

---

### Question 6 (7 points)

A trader observes that their 3-state HMM frequently switches between regimes (average duration < 5 days per regime). What might this indicate?

Select all that apply:

A) The model is overfitting with too many states
B) The transition probabilities are too low (not persistent enough)
C) The emission distributions overlap significantly
D) The data may not exhibit clear regime structure

**Answers: A, B, C, D (all are possible causes)**

**Explanation:**

**A - TRUE: Overfitting**
- Too many states for the data structure
- Model tries to explain noise as regime switches
- Solution: Try 2-state model, use BIC/AIC for model selection

**B - TRUE: Low persistence**
- If self-transition probabilities are low (e.g., 0.60), regimes don't persist
- Expected duration: E[duration] = 1/(1-p_ii)
  - p_ii = 0.60 → E[duration] = 2.5 days (very short)
  - p_ii = 0.95 → E[duration] = 20 days (persistent)
- Check estimated transition matrix

**C - TRUE: Overlapping distributions**
- If state means/variances are similar, observations don't strongly favor any state
- Leads to uncertain state inference and frequent "switches" driven by noise
- Check if states are well-separated

**D - TRUE: No true regimes**
- Data may be better modeled as a single process (e.g., GARCH)
- Regime structure is an artifact of model assumptions
- Test with simpler models

**Diagnostics:**

1. **Check state parameters:**
   - Are μ_1, μ_2, μ_3 distinct?
   - Are σ_1, σ_2, σ_3 distinct?

2. **Examine transition matrix:**
   - Are diagonal elements high (>0.90)?
   - Low diagonals → low persistence

3. **Plot regime probabilities:**
   - Smoothing P(S_t | O_{1:T}) over time
   - Frequent uncertainty (P ≈ 0.33 for all states) → poor fit

4. **Model comparison:**
   - 2-state vs 3-state BIC
   - HMM vs GARCH vs single-regime

5. **Economic validation:**
   - Do regimes correspond to known market events?
   - Are regime-based strategies profitable out-of-sample?

---

## Section 3: Trading Strategies (25 points)

### Question 7 (10 points)

A quantitative trader develops a regime-based strategy:

```
If P(S_t = Bull | O_{1:t}) > 0.7:
    - Long equity (1.2x leverage)
    - Long volatility selling (short VIX)

If P(S_t = Bear | O_{1:t}) > 0.7:
    - Reduce equity to 0.3x
    - Long volatility protection (long VIX)

If P(S_t = Neutral | O_{1:t}) > 0.7:
    - Market neutral (1.0x)
    - No volatility position

Otherwise (uncertain regime):
    - Defensive default (0.5x equity, no vol)
```

**Part A (5 points):** What is the purpose of the 0.7 probability threshold?

**Answer:**

The 0.7 threshold serves as a **confidence filter** to avoid trading on uncertain regime assignments.

**Without threshold:**
- Trade on every marginal regime shift
- P(Bull) = 0.35, P(Neutral) = 0.34, P(Bear) = 0.31 → Which regime?
- Leads to excessive trading (high transaction costs)
- Whipsaws: frequent switches in choppy markets

**With threshold:**
- Only act when regime is highly probable (≥70%)
- P(Bull) = 0.75 → Strong Bull signal (trade)
- P(Bull) = 0.50, P(Neutral) = 0.45 → Uncertain (use default)
- Reduces false signals and transaction costs

**Benefits:**
1. **Lower turnover:** Trade only high-conviction signals
2. **Better risk-adjusted returns:** Avoid whipsaws
3. **Robustness:** Less sensitive to small parameter changes
4. **Transaction costs:** Fewer trades = lower costs

**Calibration:**
- Higher threshold (0.8-0.9): Fewer trades, higher confidence
- Lower threshold (0.6): More trades, faster regime response
- Backtest to find optimal threshold for your strategy

**Part B (5 points):** Why might this strategy underperform during regime transitions?

**Answer:**

**Lag in regime detection:**

HMMs detect regime switches with a delay because:
1. **Evidence accumulation:** Need multiple observations to overcome prior state persistence
2. **Filtering uses only past data:** P(S_t | O_{1:t}) doesn't know future confirms the switch
3. **High confidence threshold:** Waiting for P(S_t) > 0.7 adds delay

**Example transition (Bull → Bear):**

```
Day  True State  P(Bear|past)  Action        Return
1    Bull        0.05          Long 1.2x     +1.0%
2    Bull        0.08          Long 1.2x     +0.5%
3    BEAR        0.15          Long 1.2x     -2.0%  (lag!)
4    BEAR        0.35          Long 1.2x     -2.5%  (lag!)
5    BEAR        0.65          Uncertain 0.5x -1.5%  (lag!)
6    BEAR        0.75          Defensive 0.3x -0.8%  (finally!)
```

**Losses during lag:**
- Days 3-5: Still long despite Bear regime
- Miss protective action when needed most
- Worst returns often occur at regime onset (crashes start suddenly)

**Mitigation strategies:**

1. **Lower threshold:** React faster (but more whipsaws)
2. **Add leading indicators:** Combine HMM with technical signals
3. **Gradual adjustment:** Reduce exposure as P(Bear) rises, not just at threshold
4. **Stop-loss overlays:** Exit on large losses regardless of regime
5. **Implied volatility:** Use option markets for early warning

**Trade-off:**
- Fast reaction → more false signals, higher costs
- Slow reaction → miss early regime changes, larger drawdowns
- Optimal threshold balances these concerns

---

### Question 8 (8 points)

Compare using the **Viterbi path** versus **filtering probabilities** for a live trading strategy.

**Approach 1: Viterbi Path**
- Decode most likely state sequence: S*_{1:t}
- Trade based on S*_t (Bull/Neutral/Bear)

**Approach 2: Filtering Probabilities**
- Compute P(S_t | O_{1:t}) for each state
- Trade based on probabilities (e.g., allocate proportionally)

Which is more appropriate for live trading and why?

**Answer: Filtering probabilities (Approach 2)**

**Comparison:**

| Aspect | Viterbi Path | Filtering Probabilities |
|--------|--------------|-------------------------|
| Output | Single state | Distribution over states |
| Uncertainty | Not captured | Explicitly quantified |
| Position sizing | Binary (full/none) | Gradual (proportional) |
| Whipsaws | More frequent | Less frequent |
| Updates | Entire path changes | Only current belief |

**Why filtering is better for trading:**

1. **Captures uncertainty:**
   - Viterbi: "State is Bull" (overconfident)
   - Filtering: "60% Bull, 30% Neutral, 10% Bear" (realistic)

2. **Gradual adjustments:**
   ```
   Viterbi: 100% equity → 0% equity (sudden)
   Filtering: 100% equity → 80% → 60% → 40% → 0% (gradual)
   ```

3. **Fewer whipsaws:**
   - Viterbi switches states discretely (all or nothing)
   - Filtering adjusts probabilities smoothly

4. **Better risk management:**
   - Position size ∝ confidence
   - If P(Bull) = 0.55 (uncertain), use moderate exposure
   - If P(Bull) = 0.95 (confident), use full exposure

5. **Stable over time:**
   - Viterbi path can change retroactively as new data arrives
   - Filtering only updates forward belief (no revision of past decisions)

**Implementation example:**

```python
# Filtering approach
P_bull = filtering_prob[BULL]
P_neutral = filtering_prob[NEUTRAL]
P_bear = filtering_prob[BEAR]

# Proportional allocation
equity_weight = 1.5*P_bull + 1.0*P_neutral + 0.3*P_bear

# Example:
# P = [0.6, 0.3, 0.1] → weight = 1.5*0.6 + 1.0*0.3 + 0.3*0.1 = 1.23
# P = [0.2, 0.3, 0.5] → weight = 1.5*0.2 + 1.0*0.3 + 0.3*0.5 = 0.75
```

**When Viterbi might be useful:**
- Backtesting (retrospective analysis with smoothing)
- Understanding historical regimes
- Academic research
- Not for live trading

---

### Question 9 (7 points)

A strategy based on HMM regime detection shows strong backtested returns but fails in live trading. List three potential reasons.

**Answer:**

**1. Look-ahead bias (smoothing instead of filtering)**

**Problem:**
- Backtest used P(S_t | O_{1:T}, λ) (smoothing with future data)
- Live trading uses P(S_t | O_{1:t}, λ) (filtering with only past)
- Smoothing is more accurate but unavailable in real-time

**Example:**
- Backtest: Knew Oct 2008 was crisis using data through Dec 2008
- Live: On Oct 15, 2008, only knew data through Oct 15 (less certain)

**Fix:** Backtest with filtering only

**2. In-sample overfitting**

**Problem:**
- Model parameters optimized on training data
- Learned patterns specific to historical period
- Don't generalize to new market conditions

**Example:**
- Trained on 2010-2020 (bull market decade)
- Learned "Bull state" parameters from that period
- Fails in 2020-2025 (different volatility regime)

**Fix:**
- Out-of-sample validation
- Rolling window estimation
- Regularization

**3. Transaction costs and slippage**

**Problem:**
- Backtest assumes perfect execution at closing prices
- Live trading incurs:
  - Bid-ask spreads
  - Market impact (slippage on large orders)
  - Commission/fees
- Frequent regime switches → high turnover → large costs

**Example:**
- Backtest: Switch from 100% equity to 0% (assume no cost)
- Live: Spread on $10M equity trade = $20k-50k (0.2-0.5%)

**Fix:**
- Model realistic transaction costs in backtest
- Use higher regime probability thresholds
- Implement gradual position adjustments

**Additional reasons:**

**4. Parameter uncertainty:**
- Estimated parameters have uncertainty
- Backtest uses single "best" estimate
- Reality has parameter drift and estimation error

**5. Market regime shift:**
- Model assumes stationary parameters
- Real markets evolve (structural breaks)
- Parameters estimated on 2000-2020 invalid for 2025

**6. Data snooping:**
- Tried many model variants (2-state, 3-state, different features)
- Selected best-performing model
- Multiple testing inflates apparent performance

---

## Section 4: Portfolio Allocation (25 points)

### Question 10 (12 points)

A multi-asset portfolio manager uses a 2-state HMM to inform allocation between stocks and bonds:

```
State 1 (Risk-On): High equity expected returns, low vol
State 2 (Risk-Off): Low/negative equity returns, high vol

Current filtering probabilities:
P(S_t = Risk-On | O_{1:t}) = 0.65
P(S_t = Risk-Off | O_{1:t}) = 0.35
```

**Part A (6 points):** Design a regime-conditional portfolio allocation rule. Specify weights for stocks and bonds in each regime and how to handle uncertainty.

**Answer:**

**Regime-specific optimal allocations:**

**State 1 (Risk-On):**
- Stocks: 70%
- Bonds: 30%
- Rationale: Higher expected equity returns, lower risk

**State 2 (Risk-Off):**
- Stocks: 20%
- Bonds: 80%
- Rationale: Capital preservation, equity downside risk

**Uncertainty-weighted allocation:**

w_t = P(Risk-On | O_{1:t}) × w_{Risk-On} + P(Risk-Off | O_{1:t}) × w_{Risk-Off}

**Current allocation:**
```
w_stocks = 0.65 × 0.70 + 0.35 × 0.20 = 0.455 + 0.070 = 0.525 (52.5%)
w_bonds = 0.65 × 0.30 + 0.35 × 0.80 = 0.195 + 0.280 = 0.475 (47.5%)
```

**Properties:**
- Gradual adjustment as probabilities change
- No discrete jumps (smooth rebalancing)
- Naturally incorporates uncertainty
- More uncertain → closer to balanced allocation

**Part B (6 points):** The manager implements the strategy above. Over the next month, P(Risk-On) gradually decreases from 0.65 to 0.30. Describe the portfolio evolution and transaction costs implications.

**Answer:**

**Portfolio evolution:**

```
Week  P(Risk-On)  Stock %  Bond %  Change
0     0.65        52.5%    47.5%   -
1     0.55        45.5%    54.5%   -7.0%
2     0.45        38.5%    61.5%   -7.0%
3     0.35        31.5%    68.5%   -7.0%
4     0.30        27.0%    73.0%   -4.5%
```

**Total change:** 52.5% → 27.0% stocks (reduced by 25.5 percentage points)

**Transaction costs:**

**Weekly rebalancing:**
- Week 1: Sell 7% stocks, buy 7% bonds
- Week 2: Sell 7% stocks, buy 7% bonds
- Week 3: Sell 7% stocks, buy 7% bonds
- Week 4: Sell 4.5% stocks, buy 4.5% bonds
- **Total turnover:** 4 × 7% + 4.5% = 32.5% one-way

On $100M portfolio:
- Turnover = $32.5M
- Costs (assume 10 bps) = $32,500

**Alternative: Monthly rebalancing**
- Single 25.5% reduction at end
- Turnover = 25.5% one-way
- Costs = $25,500
- **Savings: $7,000** (but higher path risk)

**Optimization strategies:**

1. **Tolerance bands:** Only rebalance if allocation drifts >5% from target
   - Reduces trades in range-bound probabilities
   - Lower costs, some tracking error

2. **Longer rebalancing period:** Weekly → Monthly
   - Fewer trades, lower costs
   - More tracking error to target

3. **Futures/ETFs:** Use derivatives for tactical adjustments
   - Lower transaction costs
   - Easier to scale in/out

4. **Partial rebalancing:** Move 50% toward target each period
   - Smooths execution
   - Reduces market impact

**Trade-off:** Transaction costs vs. tracking error to optimal allocation

---

### Question 11 (8 points)

Explain how incorporating regime probabilities improves upon traditional mean-variance optimization.

**Answer:**

**Traditional Mean-Variance Optimization:**

Maximize: w^T μ - (γ/2) w^T Σ w

Where:
- w: portfolio weights
- μ: expected returns (single estimate)
- Σ: covariance matrix (single estimate)
- γ: risk aversion

**Problems:**

1. **Single regime assumption:**
   - Assumes returns come from one distribution
   - Averages across all market conditions
   - μ and Σ estimated on full history (mixed regimes)

2. **Parameter uncertainty:**
   - Estimates of μ highly uncertain
   - Small estimation errors → large allocation changes
   - Optimizer amplifies estimation error

3. **No regime adaptation:**
   - Static allocation regardless of market state
   - Can't adjust for changing conditions

**Regime-Conditional Optimization:**

For each regime k, estimate:
- μ_k: regime-specific expected returns
- Σ_k: regime-specific covariance

Compute optimal weights in each regime:
w*_k = argmax [w^T μ_k - (γ/2) w^T Σ_k w]

**Final allocation:**

w_t = Σ_k P(S_t = k | O_{1:t}) × w*_k

**Advantages:**

1. **Regime-specific parameters:**
   - Bull market: μ_Bull = [0.08%, 0.02%] (stocks, bonds)
   - Bear market: μ_Bear = [-0.15%, 0.05%]
   - More accurate than averaging

2. **Dynamic adaptation:**
   - Allocation adjusts as regime probabilities change
   - Automatic risk management (reduce equity as P(Bear) rises)

3. **Better risk estimates:**
   - Σ_Bull ≠ Σ_Bear (different volatilities and correlations)
   - Captures regime-dependent risk

4. **Accounts for uncertainty:**
   - If P(Bull) = P(Bear) = 0.5 (uncertain), allocation is balanced
   - Automatically defensive when uncertain

**Example:**

**Traditional MVO:** Always 60/40 stocks/bonds

**Regime-conditional:**
- P(Bull) = 0.9 → 70/30 (aggressive)
- P(Bull) = 0.5, P(Bear) = 0.5 → 50/50 (balanced)
- P(Bear) = 0.9 → 20/80 (defensive)

**Empirical benefits:**
- Higher Sharpe ratio (better risk-adjusted returns)
- Lower maximum drawdown (tail risk management)
- More stable allocations (less turnover than raw MVO)

---

### Question 12 (5 points)

A portfolio shows the following regime-conditional Sharpe ratios:

```
Bull regime: 1.8
Neutral regime: 0.4
Bear regime: -0.5 (negative)
```

The model spends 50%, 30%, and 20% of time in each regime respectively. What is the overall Sharpe ratio?

A) (1.8 + 0.4 - 0.5) / 3 = 0.57
B) 0.5 × 1.8 + 0.3 × 0.4 + 0.2 × (-0.5) = 0.92
C) Cannot be computed from regime-conditional Sharpe ratios
D) √(0.5 × 1.8² + 0.3 × 0.4² + 0.2 × 0.5²) = 1.27

**Answer: C (Cannot be computed from regime-conditional Sharpe ratios alone)**

**Explanation:**

**Sharpe ratio:** SR = μ_p / σ_p (portfolio return / portfolio volatility)

**Why we can't simply average:**

The overall Sharpe ratio is:
SR_{overall} = μ_{overall} / σ_{overall}

Where:
- μ_{overall} = Σ_k π_k × μ_k (CAN be computed as weighted average of regime means)
- σ_{overall} = √[Σ_k π_k × σ_k² + Σ_k π_k × (μ_k - μ_{overall})²] (NOT a weighted average)

The overall volatility has two components:
1. **Within-regime volatility:** Average of regime-specific variances
2. **Between-regime volatility:** Variance due to regime switching

**Why option B is wrong:**

B computes the weighted average of Sharpe ratios:
0.5 × (μ_Bull/σ_Bull) + 0.3 × (μ_Neutral/σ_Neutral) + 0.2 × (μ_Bear/σ_Bear)

This is NOT equal to μ_overall/σ_overall because:
- Sharpe ratios are ratios (can't be averaged)
- Like averaging speeds: (60 mph + 40 mph)/2 ≠ total distance/total time

**What we'd need:**

To compute overall Sharpe, we need:
- μ_k: regime-specific mean returns
- σ_k: regime-specific volatilities
- π_k: regime probabilities

Then:
```
μ_overall = Σ π_k × μ_k
σ²_overall = Σ π_k × σ_k² + Σ π_k × (μ_k - μ_overall)²
SR_overall = μ_overall / σ_overall
```

Regime-conditional Sharpe ratios alone are insufficient.

---

## Bonus Question (5 points)

### Question 13 (5 points)

Derive the expected duration of a regime with self-transition probability p.

**Solution:**

Let D be the random variable representing the duration (number of periods) in the regime.

**Probability of staying exactly n periods:**

P(D = n) = p^{n-1} × (1-p)

- Stay for n-1 periods: probability p^{n-1}
- Leave on the n-th period: probability (1-p)

**Expected duration:**

E[D] = Σ_{n=1}^{∞} n × P(D = n) = Σ_{n=1}^{∞} n × p^{n-1} × (1-p)

Factor out (1-p):
E[D] = (1-p) × Σ_{n=1}^{∞} n × p^{n-1}

**Recognize the series:** Σ_{n=1}^{∞} n × x^{n-1} = 1/(1-x)² for |x| < 1

E[D] = (1-p) × 1/(1-p)²
     = 1/(1-p)

**Result: E[Duration] = 1/(1-p)**

**Examples:**
- p = 0.90 → E[D] = 1/0.10 = 10 periods
- p = 0.95 → E[D] = 1/0.05 = 20 periods
- p = 0.99 → E[D] = 1/0.01 = 100 periods

**Financial interpretation:**
- High persistence (p ≈ 1) → long regime duration
- Low persistence (p ≈ 0.5) → short regime duration
- For daily data with p = 0.95, expected regime lasts 20 days (≈ 1 month)

---

## Answer Key Summary

1. Part A: State labels (5 pts), Part B: State 2 most likely (5 pts)
2. B (8 pts)
3. Viterbi vs marginal explanation (7 pts)
4. Part A: Switch at t=5 or t=6 (5 pts), Part B: Use filtering (5 pts)
5. HMM vs GARCH comparison (8 pts)
6. A, B, C, D (7 pts)
7. Part A: Threshold purpose (5 pts), Part B: Lag explanation (5 pts)
8. Filtering probabilities preferred (8 pts)
9. Three reasons for live failure (7 pts)
10. Part A: Allocation rule (6 pts), Part B: Cost analysis (6 pts)
11. Regime-conditional vs traditional MVO (8 pts)
12. C (5 pts)
13. E[D] = 1/(1-p) derivation (5 pts - Bonus)

**Total: 100 points (105 with bonus)**

---

## Grading Rubric

- **90-100 points:** Excellent - Strong understanding of HMM financial applications
- **80-89 points:** Good - Solid grasp with minor gaps in practical implementation
- **70-79 points:** Satisfactory - Adequate knowledge, review trading strategies
- **60-69 points:** Needs Improvement - Review regime detection and portfolio allocation
- **Below 60:** Incomplete Understanding - Revisit all application materials

---

## Learning Objectives Assessed

- [ ] Interpret HMM regimes in financial contexts
- [ ] Distinguish filtering, smoothing, and Viterbi decoding
- [ ] Design regime-based trading strategies
- [ ] Understand lag and uncertainty in regime detection
- [ ] Compare HMM to other volatility models (GARCH)
- [ ] Apply HMM to portfolio allocation
- [ ] Diagnose overfitting and poor model fit
- [ ] Account for transaction costs in implementation
- [ ] Integrate regime probabilities into optimization
