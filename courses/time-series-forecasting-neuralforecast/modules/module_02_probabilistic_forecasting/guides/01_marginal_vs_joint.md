# Marginal vs Joint Distributions: Why Your Forecast Intervals Lie

> **Reading time:** ~16 min | **Module:** 2 — Probabilistic Forecasting | **Prerequisites:** Module 1

## In Brief

Prediction intervals describe uncertainty at a **single point in time**. They say nothing about how uncertainty unfolds across multiple periods together. This distinction — marginal versus joint distributions — is the central insight that separates good probabilistic forecasting from dangerous probabilistic forecasting.

<div class="callout-insight">
<strong>Insight:</strong> The 80th percentile of Monday's sales plus the 80th percentile of Tuesday's sales is NOT the 80th percentile of the combined two-day total. Not even close.
</div>

<div class="callout-key">
<strong>Key Concept:</strong> Prediction intervals describe uncertainty at a **single point in time**. They say nothing about how uncertainty unfolds across multiple periods together.
</div>


---

## Start Here: Code First

Train NHITS with quantile loss on the French Bakery dataset and look at what you actually get.

<div class="callout-insight">
<strong>Insight:</strong> Train NHITS with quantile loss on the French Bakery dataset and look at what you actually get.
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# ── Load French Bakery data ──────────────────────────────────────────────────
# Daily baguette sales across multiple bakery locations
url = "https://raw.githubusercontent.com/nicholasjmorales/French-Bakery-Daily-Transactional-Dataset/main/Bakery_sales.csv"
raw = pd.read_csv(url, parse_dates=["date"])

# Aggregate to one series: total daily baguette transactions
baguettes = (
    raw[raw["article"].str.upper().str.contains("BAGUETTE")]
    .groupby("date")["Quantity"]
    .sum()
    .reset_index()
    .rename(columns={"date": "ds", "Quantity": "y"})
    .assign(unique_id="bakery_total")
    .sort_values("ds")
)

# Keep complete weeks only
baguettes = baguettes[baguettes["ds"].dt.weekday < 7]
print(f"Dataset: {len(baguettes)} days of baguette sales")
print(baguettes.head(10))
```
</div>

The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
# ── Train NHITS with quantile loss ───────────────────────────────────────────
# MQLoss(level=[80, 90]) requests two symmetric prediction intervals
# This produces 4 columns: lo-80, hi-80, lo-90, hi-90

model = NHITS(
    h=7,                          # Forecast horizon: one week (7 days)
    input_size=28,                # 4 weeks of history
    loss=MQLoss(level=[80, 90]),  # Marginal quantiles at 80% and 90%
    max_steps=500,
    random_seed=42,
)

nf = NeuralForecast(models=[model], freq="D")

# Train on all but last 2 weeks
cutoff = baguettes["ds"].max() - pd.Timedelta(days=14)
train_df = baguettes[baguettes["ds"] <= cutoff]
nf.fit(df=train_df)

# Forecast the next 7 days
forecast = nf.predict()
print("\nForecast columns:", forecast.columns.tolist())
print(forecast)
```
</div>

The output columns are:
- `NHITS` — the point forecast (mean)
- `NHITS-lo-80` — lower bound of the 80% interval (10th percentile)
- `NHITS-hi-80` — upper bound of the 80% interval (90th percentile)
- `NHITS-lo-90` — lower bound of the 90% interval (5th percentile)
- `NHITS-hi-90` — upper bound of the 90% interval (95th percentile)

Each row is one day. Each interval is a **marginal** interval — it describes uncertainty for that day **in isolation**.

---

## What Marginal Distributions Capture

A marginal distribution $F_t(y_t)$ describes the probability of the outcome at a single time step $t$, regardless of what happens at any other time step.

<div class="callout-key">
<strong>Key Point:</strong> A marginal distribution $F_t(y_t)$ describes the probability of the outcome at a single time step $t$, regardless of what happens at any other time step.
</div>


$$F_t(y_t) = P(Y_t \leq y_t)$$

The 80th percentile $q_t^{0.8}$ satisfies:

$$P(Y_t \leq q_t^{0.8}) = 0.80$$

This is what NHITS produces for each day. It is correct and useful. For a single-day decision — "should I order enough inventory to cover today's demand at 80% service level?" — this is exactly what you need.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# ── Visualize the marginal prediction intervals ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.fill_between(
    forecast["ds"],
    forecast["NHITS-lo-90"],
    forecast["NHITS-hi-90"],
    alpha=0.25, color="steelblue", label="90% interval"
)
ax.fill_between(
    forecast["ds"],
    forecast["NHITS-lo-80"],
    forecast["NHITS-hi-80"],
    alpha=0.4, color="steelblue", label="80% interval"
)
ax.plot(forecast["ds"], forecast["NHITS"], color="steelblue",
        linewidth=2, label="Point forecast")

# Plot actuals from test period
test_df = baguettes[baguettes["ds"] > cutoff]
ax.plot(test_df["ds"], test_df["y"], "k.", markersize=6, label="Actual")

ax.set_title("NHITS with MQLoss: Marginal Prediction Intervals (7-day horizon)")
ax.set_xlabel("Date")
ax.set_ylabel("Baguettes sold")
ax.legend()
plt.tight_layout()
plt.show()
```
</div>

The fan chart looks informative. Each day has a plausible range. But there is a hidden problem.

---

## The Problem: What Marginal Distributions Miss

Consider this business question:

<div class="callout-info">
<strong>Info:</strong> Consider this business question:

> "How many total baguettes should I order for the entire week to meet demand with 80% probability?"

A natural (wrong) answer: sum up the 80th percentile for each da...
</div>


> "How many total baguettes should I order for the entire week to meet demand with 80% probability?"

A natural (wrong) answer: sum up the 80th percentile for each day.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# ── The naive (wrong) approach ────────────────────────────────────────────────
weekly_80_naive = forecast["NHITS-hi-80"].sum()
weekly_point = forecast["NHITS"].sum()

print(f"Sum of daily 80th percentiles: {weekly_80_naive:.0f} baguettes")
print(f"Sum of daily point forecasts:  {weekly_point:.0f} baguettes")
print(f"Naive 'buffer':                {weekly_80_naive - weekly_point:.0f} baguettes")
```
</div>

This gives a specific number. It looks precise. It is wrong.

To see why, we need to understand the difference between **marginal** and **joint** distributions.

---

## Marginal vs Joint: The Core Distinction

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
graph TD
    subgraph Marginal["Marginal Distributions (what NHITS gives us)"]
        M1["P(Y_Mon ≤ q) = 0.80"]
        M2["P(Y_Tue ≤ q) = 0.80"]
        M3["P(Y_Wed ≤ q) = 0.80"]
        M7["P(Y_Sun ≤ q) = 0.80"]
        M1 -.- M2 -.- M3 -.- M7
        note1["Each day is independent\nNo cross-day correlation"]
    end

<div class="callout-key">
<strong>Key Point:</strong> The **joint distribution** $F_{1:H}(y_1, \ldots, y_H)$ describes the probability of all outcomes together:

$$F_{1:H}(y_1, \ldots, y_H) = P(Y_1 \leq y_1, Y_2 \leq y_2, \ldots, Y_H \leq y_H)$$

When yo...
</div>


    subgraph Joint["Joint Distribution (what we need)"]
        J["P(Y_Mon ≤ q₁, Y_Tue ≤ q₂, ..., Y_Sun ≤ q₇) = 0.80"]
        note2["All 7 days together\nCaptures temporal correlation"]
    end

    Marginal -->|"marginalization\n(loses correlation)"| Joint
```

The **joint distribution** $F_{1:H}(y_1, \ldots, y_H)$ describes the probability of all outcomes together:

$$F_{1:H}(y_1, \ldots, y_H) = P(Y_1 \leq y_1, Y_2 \leq y_2, \ldots, Y_H \leq y_H)$$

When you sum across a horizon, you need the joint distribution of the sum:

$$S = Y_1 + Y_2 + \cdots + Y_H$$

The 80th percentile of $S$ is the value $q^{0.8}_S$ such that:

$$P(S \leq q^{0.8}_S) = 0.80$$

**This is NOT the sum of individual 80th percentiles**, unless the days are perfectly correlated.

---


<div class="compare">
<div class="compare-card">
<div class="header before">Marginal</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Joint: The Core Distinction</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## The Key Formula: Why Sums Break

For the sum of $H$ random variables:

<div class="callout-key">
<strong>Key Point:</strong> For the sum of $H$ random variables:

$$q^{\alpha}(Y_1 + \cdots + Y_H) \neq q^{\alpha}(Y_1) + \cdots + q^{\alpha}(Y_H)$$

The left side is the $\alpha$-quantile of the total.
</div>


$$q^{\alpha}(Y_1 + \cdots + Y_H) \neq q^{\alpha}(Y_1) + \cdots + q^{\alpha}(Y_H)$$

The left side is the $\alpha$-quantile of the total. The right side is the sum of $\alpha$-quantiles. They are equal only under perfect positive correlation (all variables move together).

In general, if the daily outcomes have some independence:

$$q^{0.8}(S) < \sum_{t=1}^{H} q^{0.8}(Y_t)$$

The sum of 80th percentiles **overestimates** the 80th percentile of the sum. By using the sum of marginals, you would order too much inventory.

Worse: if you try to compute, say, the probability that total weekly demand exceeds a threshold, you simply cannot do it from marginal quantiles alone. You need the joint structure.

---

## Concrete Demonstration: The Numbers Don't Lie

```python
# ── Simulate the failure with real-world numbers ──────────────────────────────
np.random.seed(42)

# Realistic bakery parameters from the data
daily_mean = baguettes["y"].mean()
daily_std = baguettes["y"].std()

n_simulations = 50_000
n_days = 7

# Case 1: Independent days (zero temporal correlation)
# This is what MQLoss implicitly assumes between time steps
independent_days = np.random.normal(
    loc=daily_mean,
    scale=daily_std,
    size=(n_simulations, n_days)
)
weekly_totals_independent = independent_days.sum(axis=1)

# Case 2: Highly correlated days (e.g., weather or event drives the whole week)
# A common shock plus individual noise
common_shock = np.random.normal(0, daily_std * 0.7, size=(n_simulations, 1))
individual_noise = np.random.normal(0, daily_std * 0.3, size=(n_simulations, n_days))
correlated_days = daily_mean + common_shock + individual_noise
weekly_totals_correlated = correlated_days.sum(axis=1)

# Per-day 80th percentile
per_day_q80 = np.percentile(independent_days, 80, axis=0)
sum_of_q80 = per_day_q80.sum()

# True 80th percentile of the weekly total
true_q80_independent = np.percentile(weekly_totals_independent, 80)
true_q80_correlated = np.percentile(weekly_totals_correlated, 80)

print("=" * 55)
print("THE SUM OF QUANTILES ≠ QUANTILE OF SUMS")
print("=" * 55)
print(f"\nSum of daily 80th percentiles:        {sum_of_q80:,.0f}")
print(f"\nTrue 80th percentile of weekly total:")
print(f"  Independent days:                   {true_q80_independent:,.0f}")
print(f"  Correlated days:                    {true_q80_correlated:,.0f}")
print(f"\nError from using sum-of-quantiles:")
print(f"  vs independent: +{sum_of_q80 - true_q80_independent:,.0f} baguettes "
      f"({(sum_of_q80/true_q80_independent - 1)*100:.1f}% overestimate)")
print(f"  vs correlated:  +{sum_of_q80 - true_q80_correlated:,.0f} baguettes "
      f"({(sum_of_q80/true_q80_correlated - 1)*100:.1f}% overestimate)")
```

```python
# ── Visualize the distributions to make the gap visible ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, totals, label, color in zip(
    axes,
    [weekly_totals_independent, weekly_totals_correlated],
    ["Independent days", "Correlated days"],
    ["steelblue", "darkorange"],
):
    ax.hist(totals, bins=80, alpha=0.7, color=color, density=True)
    true_q80 = np.percentile(totals, 80)
    ax.axvline(true_q80, color="darkgreen", linewidth=2.5,
               label=f"True 80th pct: {true_q80:,.0f}")
    ax.axvline(sum_of_q80, color="crimson", linewidth=2.5, linestyle="--",
               label=f"Sum of daily 80th pcts: {sum_of_q80:,.0f}")
    ax.set_title(f"Weekly Total Distribution\n({label})")
    ax.set_xlabel("Total baguettes")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.suptitle("Why Sum of Quantiles ≠ Quantile of Sums", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

The red dashed line (sum of daily 80th percentiles) always falls to the right of the green line (true 80th percentile of the weekly total). The gap depends on the correlation structure — but it is always non-zero unless days are perfectly correlated.

---

## Three Business Failure Scenarios

### Scenario 1: Annual Budget Forecasting

A company uses monthly marginal 80% intervals to set its annual budget. They sum the upper bounds for all 12 months, believing they have an 80% confidence budget.

**Reality:** The probability that the sum of 12 monthly actuals fits within this budget is much higher than 80% (because the errors partially cancel out — overestimating in one month offsets underestimating in another). They have over-budgeted. Capital that could have been deployed is sitting idle.

### Scenario 2: Inventory Reorder Timing

A warehouse manager checks whether cumulative demand from now through the reorder lead time will exceed the safety stock. She pulls the 95th percentile for each day and sums them.

**Reality:** This sum overstates the true 95th percentile of cumulative demand. She sets safety stock too high, tying up working capital unnecessarily.

### Scenario 3: Multi-Period Order Sizing

A bakery owner needs enough flour to last one week at 90% service level. She sums the 90th percentile daily requirements.

**Reality:** If daily demand is even mildly mean-reverting (a high day tends to be followed by an average day), the sum of marginals dramatically overestimates what she needs. She over-orders flour, increases waste, and reduces margin.

---

## The Mathematical Root Cause

Consider two periods with outcomes $Y_1, Y_2$ and their sum $S = Y_1 + Y_2$.

If $Y_1$ and $Y_2$ are independent with mean $\mu$ and standard deviation $\sigma$:

$$\text{Var}(S) = \text{Var}(Y_1) + \text{Var}(Y_2) = 2\sigma^2$$
$$\text{SD}(S) = \sigma\sqrt{2} \approx 1.41\sigma$$

If we naively sum the 80th percentiles (where the 80th percentile is at $\mu + 0.84\sigma$):

$$q^{0.8}(Y_1) + q^{0.8}(Y_2) = 2(\mu + 0.84\sigma) = 2\mu + 1.68\sigma$$

The true 80th percentile of the sum (assuming normality):

$$q^{0.8}(S) = 2\mu + 0.84 \cdot \sigma\sqrt{2} = 2\mu + 1.19\sigma$$

The error: $1.68\sigma - 1.19\sigma = 0.49\sigma$ per period, growing as $\sqrt{H}$ for longer horizons.

For a 7-day horizon:
- Sum of marginals contributes $7 \times 0.84\sigma = 5.88\sigma$ above the mean
- True joint 80th percentile contributes $0.84\sigma\sqrt{7} = 2.22\sigma$ above the mean
- Overestimate: $5.88\sigma - 2.22\sigma = 3.66\sigma$

**The error grows with the horizon. The longer your planning window, the more wrong marginal quantiles become.**

---

## What We Actually Need: Joint Distributions via Sample Paths

To answer multi-period questions correctly, we need the **joint distribution** of outcomes across the entire horizon. The cleanest way to represent a joint distribution is through **sample paths** — simulated trajectories that preserve temporal correlation.

```python
# ── Preview: what sample paths look like ─────────────────────────────────────
# (Detailed in Module 3 — this is just the teaser)
np.random.seed(0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

days = range(7)

# Left: marginal intervals (what we have now)
ax = axes[0]
for day in days:
    q10 = np.percentile(independent_days[:, day], 10)
    q90 = np.percentile(independent_days[:, day], 90)
    q20 = np.percentile(independent_days[:, day], 20)
    q80 = np.percentile(independent_days[:, day], 80)
    ax.vlines(day, q10, q90, colors="steelblue", linewidth=8, alpha=0.3)
    ax.vlines(day, q20, q80, colors="steelblue", linewidth=8, alpha=0.5)
    ax.plot(day, np.mean(independent_days[:, day]), "o", color="steelblue")

ax.set_title("Marginal Intervals\n(independent per day — miss correlation)")
ax.set_xlabel("Day")
ax.set_ylabel("Baguettes")
ax.set_xticks(range(7))
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# Right: sample paths (coherent trajectories)
ax = axes[1]
for i in range(30):
    ax.plot(days, independent_days[i], alpha=0.2, color="darkorange", linewidth=1)
ax.plot(days, independent_days[:30].mean(axis=0), color="darkorange",
        linewidth=2.5, label="Mean path")
ax.set_title("Sample Paths\n(coherent trajectories — preserve correlation)")
ax.set_xlabel("Day")
ax.set_ylabel("Baguettes")
ax.set_xticks(range(7))
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.legend()

plt.suptitle("From Marginal Intervals to Sample Paths", fontsize=13)
plt.tight_layout()
plt.show()

print("\nWith sample paths, computing the true 80th percentile of weekly total is trivial:")
print(f"  np.percentile(paths.sum(axis=1), 80) = {np.percentile(independent_days[:100].sum(axis=1), 80):,.0f}")
print("No distributional assumptions. No error from summing quantiles.")
```

Each sample path is a **coherent trajectory** — a plausible week of demand that respects the temporal correlation structure. To find the 80th percentile of the weekly total, you just simulate many paths, sum each one, and take the 80th percentile of those sums. No math required.

---

## From Point to Probabilistic: The Progression

```
Point forecast       →  Single number: "Sales will be 320 baguettes"
                         Can't describe uncertainty at all.

Prediction interval  →  A range per day: [280, 360] with 80% coverage
                         Describes per-day uncertainty, NOT multi-day joint uncertainty.

Marginal quantiles   →  Full marginal distribution per day: F_t(y)
                         Better, but still per-day independent. Wrong for aggregations.

Joint distribution   →  F_{1:H}(y_1,...,y_H): all days together
                         Correct for any multi-day decision. Hard to represent directly.

Sample paths         →  N simulated trajectories, each a plausible week
                         Empirical approximation of the joint distribution.
                         Easy to aggregate, analyze, and use for decisions.
```

Marginal quantiles (what MQLoss produces) sit at step 3. They are necessary — you cannot generate sample paths without first understanding the per-step marginal distributions. But they are not sufficient for multi-step business decisions.

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

### Builds on
- Module 1: Point forecasting with NHITS — understanding what the model produces
- Basic probability: marginal vs conditional distributions

### Leads to
- **Guide 02:** How MQLoss trains these marginal quantiles in detail (pinball loss)
- **Module 3:** Generating sample paths that fix the multi-step problem
- **Module 3 Notebook:** The bakery problem solved correctly with ConformalIntervals

---

## Key Takeaways

1. **Marginal quantiles describe per-step uncertainty** — they are correct for single-step decisions.
2. **Multi-step decisions require joint distributions** — the sum of marginals is not the marginal of sums.
3. **The error grows with the horizon** — a 7-day horizon has roughly $3.66\sigma$ of bias at the 80th percentile.
4. **Three failure modes:** annual budgeting, inventory reorder timing, order sizing — all stem from the same root cause.
5. **Sample paths are the solution** — coherent trajectories that capture temporal correlation and support any aggregation query.


<div class="flow">
<div class="flow-step mint">1. Marginal quantiles describe pe...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Multi-step decisions require j...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. The error grows with the horiz...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Three failure modes:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. Sample paths are the solution</div>
</div>

---

*"The sum of the quantiles is not the quantile of the sum. This is not a rounding error. It is a structural error that grows with your planning horizon."*


---

## Cross-References

<a class="link-card" href="./01_marginal_vs_joint.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_quantiles_not_enough.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
