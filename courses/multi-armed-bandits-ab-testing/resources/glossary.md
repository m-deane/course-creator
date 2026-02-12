# Glossary: Multi-Armed Bandits & A/B Testing

## Core Bandit Concepts

**Arm**
A choice or action available to the decision-maker. In commodity trading, each commodity (WTI, Gold, Copper) is an arm. In content creation, each topic-format combination is an arm.

**Reward**
The outcome observed after pulling an arm. Can be binary (success/failure), continuous (return, engagement), or risk-adjusted (Sharpe ratio). The bandit's goal is to maximize cumulative reward.

**Pull**
The act of selecting and trying an arm. Also called "playing" an arm or making a decision.

**Horizon (T)**
The total number of rounds or decisions. Can be finite (known) or infinite (unknown). Finite horizons encourage more exploration early; infinite horizons balance exploration-exploitation continuously.

**Policy (π)**
A strategy for selecting arms. Maps history and context to arm selections. Examples: epsilon-greedy, UCB, Thompson Sampling.

**Regret (R)**
The difference between your cumulative reward and what you would have gotten by always picking the best arm. Formal: `R(T) = T·μ* - Σ r_t`, where `μ*` is the best arm's mean reward.

**Expected Regret**
Average regret over random realizations of rewards. Used to compare policies theoretically.

**Cumulative Regret**
Total regret accumulated up to round T. Good policies achieve `O(log T)` regret — logarithmic growth means learning slows over time.

**Simple Regret**
Regret from the final recommendation (not the whole sequence). Relevant when exploration phase ends and you must commit to one arm.

## Explore-Exploit Tradeoff

**Exploration**
Trying arms to learn their rewards. Essential early on when uncertain, but costly if done excessively.

**Exploitation**
Choosing the empirically best arm to maximize immediate reward. Optimal in the short term, but risks missing better options.

**Explore-Exploit Dilemma**
The fundamental tension: explore to learn better options, or exploit current knowledge? No perfect solution — bandits balance this adaptively.

**Optimism in the Face of Uncertainty (OFU)**
Principle: when uncertain, assume arms might be good. Drives UCB algorithm — pick arms with high upper confidence bounds.

**Thompson Sampling / Probability Matching**
Alternative principle: allocate trials proportional to probability each arm is best. Bayesian approach that naturally balances exploration-exploitation.

## Classic Bandit Algorithms

**Epsilon-Greedy (ε-greedy)**
With probability ε, explore (pick random arm); with probability 1-ε, exploit (pick empirically best arm). Simple but requires tuning ε.

**Epsilon-Decreasing**
Start with high ε (e.g., 0.5) and gradually reduce (e.g., ε = 1/t). Explores more early, exploits more later.

**Upper Confidence Bound (UCB)**
Pick arm with highest upper confidence bound: `μ̂ + √(2 log t / n)`. Balances estimated reward and uncertainty. Achieves optimal `O(log T)` regret.

**UCB1**
Specific UCB variant with constant c=2 in confidence term. Simple, no parameters to tune, provably optimal.

**Thompson Sampling**
Maintain Bayesian posterior over each arm's reward distribution. Sample from posteriors, pick best sample. Empirically excellent, matches UCB regret.

**Softmax / Boltzmann Exploration**
Pick arms probabilistically: `P(arm i) ∝ exp(μ̂_i / τ)`. Temperature τ controls exploration. Smooth but requires tuning τ.

## Bayesian Bandits

**Prior Distribution**
Initial belief about arm rewards before seeing data. Common: Beta(1,1) for Bernoulli, Normal(0, σ²) for continuous.

**Posterior Distribution**
Updated belief after observing data. Combines prior and likelihood via Bayes' rule.

**Conjugate Prior**
Prior-likelihood pair that yields a posterior in the same family. Examples: Beta-Bernoulli, Normal-Normal, Gamma-Poisson. Makes updating simple.

**Beta Distribution (Beta(α, β))**
Distribution over probabilities [0,1]. α = successes, β = failures. Mean = α/(α+β). Used for Bernoulli bandits.

**Posterior Updating**
For Beta-Bernoulli: observe reward r ∈ {0,1}, update `α ← α + r`, `β ← β + (1-r)`. For Normal-Normal: weighted average of prior and observation.

**Credible Interval**
Bayesian confidence interval. For Beta(α, β), 95% credible interval captures range of likely success probabilities.

## Contextual Bandits

**Context (x)**
Observable features of the environment when making a decision. Examples: volatility regime, time of day, user demographics. Allows personalization.

**Contextual Bandit**
Bandit where reward depends on both arm and context: `r ~ p(r | arm, context)`. Policy maps context to arm selection.

**LinUCB (Linear UCB)**
Assumes reward is linear in context: `r = θ_arm^T x + noise`. Maintains linear model per arm, picks arm with highest upper confidence bound.

**LinTS (Linear Thompson Sampling)**
Thompson Sampling with linear model. Sample `θ̂_arm ~ N(θ_arm, Σ_arm)`, predict `r = θ̂^T x`, pick best.

**Feature Engineering**
Designing context features. For commodities: volatility, momentum, seasonality, term structure. Quality features improve learning.

## Non-Stationary Bandits

**Non-Stationary Environment**
Arm rewards change over time. Common in real markets: regimes shift, trends reverse, seasonality cycles.

**Restless Bandit**
Bandit where arms evolve even when not pulled. Example: commodities whose volatility changes whether or not you trade them.

**Discounted Thompson Sampling**
Apply exponential decay to past observations: `α ← γ·α`, `β ← γ·β` for some γ < 1. Recent data counts more than old data.

**Sliding Window**
Only use last N observations for estimation. Forgets old data completely. Simple but effective for non-stationarity.

**Change Detection**
Detect when reward distribution shifts, then reset beliefs. Methods: CUSUM, Page-Hinckley test, Bayesian change point detection.

**Regime**
A period where reward distributions are stable. Markets transition between regimes (trending, mean-reverting, high-vol, low-vol).

## Commodity-Specific Terms

**Contango**
Futures prices higher than spot prices. Forward curve slopes upward. Can indicate storage costs or expected future scarcity.

**Backwardation**
Spot price higher than futures prices. Forward curve slopes downward. Often indicates immediate scarcity or high convenience yield.

**Term Structure**
Shape of futures prices across expiration dates. Steep contango = strong carry cost. Steep backwardation = strong demand now.

**Convenience Yield**
Benefit of holding physical commodity vs futures. High when shortages expected. Causes backwardation.

**Basis**
Difference between spot and futures price: `Basis = Spot - Futures`. Basis narrows as expiration approaches (convergence).

**Roll Yield**
Return from rolling futures contracts. Positive in backwardation (buy low, sell high). Negative in contango (buy high, sell low).

**Realized Volatility**
Historical volatility computed from observed returns. Often 20-day or 60-day. Used for regime detection and risk management.

**Volatility Regime**
Classification of current volatility level: low, medium, high. Contextual feature for bandit. Different strategies work in different regimes.

**Trend Signal**
Indicator of price direction: momentum, moving average crossover, ADX. Commodities exhibit both trending and mean-reverting periods.

**Seasonality**
Predictable patterns tied to calendar. Examples: natural gas peaks in winter, grains peak in summer/fall. Can be used as context.

## Commodity Trading Applications

**Two-Wallet Framework**
Portfolio allocation: core wallet (stable, diversified) + bandit sleeve (adaptive, tilts). Balances safety and adaptation. Typical split: 80-20.

**Bandit Sleeve**
The adaptive portion of portfolio where bandit chooses allocation. Smaller than core to limit risk. Rebalanced more frequently (weekly vs monthly).

**Core Wallet**
Stable allocation maintained regardless of bandit learning. Ensures diversification and prevents disasters. Rebalanced infrequently (monthly/quarterly).

**Guardrails**
Safety constraints on bandit: position limits, minimum allocation, tilt speed limits, drawdown circuit breaker. Prevent self-sabotage.

**Position Limit**
Maximum allocation to any single arm. Prevents over-concentration. Typical: 40-50% of bandit sleeve, 20-30% of total portfolio.

**Tilt Speed Limit**
Maximum change in allocation per period. Prevents whipsaw trading. Typical: 15-20% weekly change limit.

**Circuit Breaker**
Safety mechanism that halts bandit when losses exceed threshold. Reverts to safe allocation (equal-weight core). Resets when recovered.

## Risk-Adjusted Rewards

**Sharpe Ratio**
Risk-adjusted return: `Sharpe = (return - risk_free_rate) / volatility`. Penalizes volatility. Better reward for bandits than raw return.

**Sortino Ratio**
Like Sharpe but only penalizes downside volatility. More appropriate for commodities with skewed returns.

**Maximum Drawdown**
Largest peak-to-trough decline. Measure of worst-case loss. Important constraint for commodity allocators.

**Value at Risk (VaR)**
Maximum expected loss at confidence level (e.g., 95% VaR = worst loss in 95% of scenarios). Used for risk limits.

**Expected Shortfall (CVaR)**
Expected loss when loss exceeds VaR. Captures tail risk better than VaR. Also called Conditional VaR.

## Theoretical Concepts

**Lai-Robbins Lower Bound**
Any consistent policy must have `Ω(log T)` expected regret. Logarithmic regret is optimal — you can't do better asymptotically.

**Minimax Regret**
Worst-case regret over all possible problem instances. UCB achieves minimax-optimal regret up to constants.

**Gittins Index**
For each arm, the value at which you'd be indifferent between pulling it and a known-reward alternative. Optimal for infinite-horizon discounted bandits.

**Information Theory**
Bandits perform information-directed sampling — balance immediate reward and information gain about arm rewards.

**Multi-Armed Bandit (MAB)**
Generic term for sequential decision problems with multiple arms and unknown rewards. Simpler than full reinforcement learning (no state transitions).

## Production & Deployment

**Offline Evaluation**
Testing policy on logged historical data before deploying. Methods: replay, inverse propensity scoring, doubly robust estimation.

**Counterfactual Analysis**
"What would have happened if we'd used policy π instead?" Estimate rewards of alternative policies from logged data.

**A/B Testing**
Traditional approach: split traffic 50/50, run for fixed duration, pick winner. Inefficient compared to bandits but simpler to analyze.

**Bandit vs A/B Test**
Bandits adapt during testing, minimizing regret. A/B tests are fixed, wasting samples on losers. Bandits win when costs matter and you can't stop operating.

**Logging Policy**
The policy that generated historical data. Needed for offline evaluation — you can only learn about arms that were tried.

**Propensity Score**
Probability that logging policy selected each arm. Used to debias offline evaluation (inverse propensity weighting).

**Monitoring**
Tracking bandit performance: prediction accuracy, regret, guardrail activations, regime transitions. Essential for production deployment.

**Alerting**
Automated notifications when anomalies detected: unusual allocations, poor performance, guardrail violations, data issues.

---

**Tip:** Bookmark this glossary. Bandit terminology can be dense — refer back as you encounter terms in the course.
