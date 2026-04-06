---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Probabilistic Forecasting
## Why Quantiles Aren't Enough

**Module 02 — Probabilistic Forecasting**

> The 80th percentile of Monday plus the 80th percentile of Tuesday is NOT the 80th percentile of the two-day total.

<!--
Speaker notes: This is the pivotal insight of the course. Before this module, students have been training NHITS and producing prediction intervals. This module exposes a structural flaw in how those intervals are used for multi-step decisions. The goal: build a visceral understanding of why marginal quantiles fail, so that sample paths (Module 3) feel necessary, not optional.
-->

<!-- Speaker notes: Cover the key points on this slide about Probabilistic Forecasting. Pause for questions if the audience seems uncertain. -->

---

# What We've Built So Far

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
model = NHITS(
    h=7,
    input_size=28,
    loss=MQLoss(level=[80, 90]),
)
nf.fit(df=train_df)
forecast = nf.predict()
```
</div>

Output columns: `NHITS-lo-80`, `NHITS-hi-80`, `NHITS-lo-90`, `NHITS-hi-90`

These look like everything you need.

**They are not.**

<!--
Speaker notes: Start by grounding students in what they already know. They've trained NHITS with MQLoss and seen fan charts. Now we reveal the limitation. The key is not that MQLoss is wrong — it is that the output is being misinterpreted when used for multi-day decisions.
-->


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: Cover the key points on this slide about What We've Built So Far. Pause for questions if the audience seems uncertain. -->

---

# The Bakery Question

**Single-day question:**
> "Should I order enough inventory to cover today's demand at 80% service level?"

Use the per-day 80th percentile. **Correct.**


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: Cover the key points on this slide about The Bakery Question. Pause for questions if the audience seems uncertain. -->

---

**Multi-day question:**
> "How many total baguettes should I order for the entire week at 80% service level?"

Sum the per-day 80th percentiles? **Wrong.**

<!--
Speaker notes: The distinction between these two questions is everything. Single-day decisions are exactly what marginal quantiles are designed for. Multi-day decisions require the joint distribution of all seven days together. The rest of this deck explains why these are different, and how large the error is.
-->


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: Cover the key points on this slide about slide 4. Pause for questions if the audience seems uncertain. -->

---

<!-- _class: lead -->

# Marginal vs Joint Distributions

<!--
Speaker notes: Transition into the mathematical distinction. Keep this conceptual — the formulas matter but the intuition matters more.
-->

<!-- Speaker notes: Cover the key points on this slide about Marginal vs Joint Distributions. Pause for questions if the audience seems uncertain. -->

---

# Marginal Distribution

A **marginal distribution** describes one time step at a time:

$$F_t(y_t) = P(Y_t \leq y_t)$$

The 80th percentile $q_t^{0.8}$ satisfies:

$$P\!\left(Y_t \leq q_t^{0.8}\right) = 0.80$$

MQLoss trains the model to output $q_t^{0.8}$ for each day $t$ **independently**.

> Each day is described in isolation. The co-movement across days is ignored.

<!--
Speaker notes: Emphasize "in isolation." The model sees each day's marginal distribution correctly. What it does NOT capture is whether a high-demand Monday predicts a high-demand Tuesday, or whether they are independent, or even negatively correlated. That temporal correlation structure is lost.
-->


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: Cover the key points on this slide about Marginal Distribution. Pause for questions if the audience seems uncertain. -->

---

# Joint Distribution

A **joint distribution** describes all time steps together:

$$F_{1:H}(y_1, \ldots, y_H) = P(Y_1 \leq y_1, \ldots, Y_H \leq y_H)$$

For the weekly total $S = Y_1 + \cdots + Y_7$, the 80th percentile satisfies:

$$P(S \leq q_S^{0.8}) = 0.80$$

**This requires knowing how the days co-vary.**

MQLoss does not give you this. Each day's quantile is independent of the others.

<!--
Speaker notes: The joint distribution contains everything. You can answer any question about any combination of days if you have the joint. The marginals are projections — they lose the correlation structure. This is the core mathematical issue.
-->

<!-- Speaker notes: Cover the key points on this slide about Joint Distribution. Pause for questions if the audience seems uncertain. -->

---

# The Key Inequality

$$\boxed{q^{0.8}(Y_1 + \cdots + Y_7) \neq q^{0.8}(Y_1) + \cdots + q^{0.8}(Y_7)}$$

<div class="columns">

<div>

**Left side**
True 80th percentile of weekly total

Requires the joint distribution

</div>

<div>

**Right side**
Sum of daily 80th percentiles

Easy to compute from marginals

**Always overestimates the left side**
(unless days are perfectly correlated)

</div>

</div>

<!--
Speaker notes: Write this inequality slowly. Let it land. It is surprising to many people. The intuition: if Monday demand is high (above its 80th pct), Tuesday might still be low. They partially cancel. The true weekly 80th pct is lower than summing the daily 80th pcts because errors partially cancel across independent days.
-->

<!-- Speaker notes: Cover the key points on this slide about The Key Inequality. Pause for questions if the audience seems uncertain. -->

---

# How Large Is the Error?

For independent days with standard deviation $\sigma$, horizon $H$:

| Horizon | Sum of marginals | True 80th pct | Overestimate |
|---------|-----------------|---------------|--------------|
| $H = 2$ | $2\mu + 1.68\sigma$ | $2\mu + 1.19\sigma$ | $0.49\sigma$ |
| $H = 7$ | $7\mu + 5.88\sigma$ | $7\mu + 2.22\sigma$ | $3.66\sigma$ |
| $H = 30$ | $30\mu + 25.2\sigma$ | $30\mu + 4.60\sigma$ | $20.6\sigma$ |

For a bakery where $\sigma \approx 40$ baguettes/day:
- A 7-day horizon error: $3.66 \times 40 \approx \mathbf{146}$ **extra baguettes**
- A 30-day horizon error: $20.6 \times 40 \approx \mathbf{824}$ **extra baguettes**

**The error grows with $\sqrt{H}$. Longer horizons get proportionally worse.**

<!--
Speaker notes: These numbers make the abstract concrete. 146 extra baguettes per week ordered unnecessarily. Over 52 weeks, that's 7,500 baguettes — real waste, real cost. Now multiply by larger businesses with higher volumes. The stakes are significant. Derive the math briefly: variance of sum = H*σ² under independence, so SD of sum = σ√H. The 80th pct is at 0.84 SDs above mean.
-->

<!-- Speaker notes: Cover the key points on this slide about How Large Is the Error?. Pause for questions if the audience seems uncertain. -->

---

# Visual: Marginal Intervals vs Sample Paths

```
Marginal intervals (what we have):

Mon    ████████████████   90%
       ████████████       80%
       ●                  mean

Tue    ████████████████   90%  (independent of Mon)
       ████████████       80%
       ●

...each day has no memory of the others...

Sample paths (what we need):

     Mon  Tue  Wed  Thu  Fri  Sat  Sun
─── ●────●────●────●────●────●────● path 1 (high week)
──── ●───●────●────●────●────●───● path 2 (average week)
──── ●──●─────●───●─────●────●──● path 3 (low week)
```

Sample paths carry temporal correlation. Sum each path to get weekly totals.

<!--
Speaker notes: The ASCII diagram conveys the key visual. Marginal intervals are vertical — each day independent, no information about co-movement. Sample paths are horizontal — coherent trajectories through time. To get the weekly total distribution, you sum each path (a single number) and collect all those sums.
-->

<!-- Speaker notes: Cover the key points on this slide about Visual: Marginal Intervals vs Sample Paths. Pause for questions if the audience seems uncertain. -->

---

<!-- _class: lead -->

# Three Business Failure Scenarios

<!--
Speaker notes: Now we make this concrete with business cases. These are the scenarios from the minimizeregret.com blog post by Ethan Rosenthal. Each one represents a real way that using marginal quantiles for multi-period decisions leads to poor outcomes.
-->

<!-- Speaker notes: Cover the key points on this slide about Three Business Failure Scenarios. Pause for questions if the audience seems uncertain. -->

---

# Scenario 1: Annual Budget

**Setup:** A company forecasts monthly costs with 80% prediction intervals. They set the annual budget as the sum of 12 monthly upper bounds.

**Belief:** "We have an 80% budget."

**Reality:** If months are independent, the probability that the annual total fits within this budget is **much higher than 80%** — closer to 99%+ for 12 months.

**Business cost:** Over-budgeting. Capital sitting idle instead of being invested.

> Using the sum of monthly 80th percentiles is not an 80% annual budget. It is a near-certainty annual budget that wastes capital.

<!--
Speaker notes: This is the most common failure mode in corporate finance. Budget processes typically sum monthly forecasts, treating each month's 80th or 95th percentile as independent. They end up wildly over-budgeted because errors cancel across months. The correct approach: simulate paths, sum them, take the 80th percentile of the annual totals.
-->

<!-- Speaker notes: Cover the key points on this slide about Scenario 1: Annual Budget. Pause for questions if the audience seems uncertain. -->

---

# Scenario 2: Inventory Reorder Timing

**Setup:** A warehouse tracks cumulative demand over a 14-day lead time. Manager sums the 95th percentile for each day to set safety stock.

**Belief:** "My safety stock covers the 95th percentile of cumulative demand."

**Reality:** The 95th percentile of 14-day cumulative demand is far less than the sum of 14 daily 95th percentiles.

**Business cost:** Safety stock set 60–80% too high. Working capital locked in inventory unnecessarily.

> The error is proportional to $\sqrt{H}$. A 14-day lead time amplifies the mistake by $\sqrt{14} \approx 3.7\times$ compared to a 1-day horizon.

<!--
Speaker notes: Supply chain is where this failure is most financially damaging. Safety stock calculations are textbook examples of where the temporal correlation structure matters. If demand on day 1 is high, is day 2 likely to be high or low? If they're independent, you need far less safety stock than the sum of 95th percentiles implies.
-->

<!-- Speaker notes: Cover the key points on this slide about Scenario 2: Inventory Reorder Timing. Pause for questions if the audience seems uncertain. -->

---

# Scenario 3: Multi-Period Order Sizing

**Setup:** Bakery orders flour weekly. Owner needs enough flour for one week at 90% service level. She sums the 90th percentile of daily flour requirements.

**Belief:** "This order ensures 90% probability of not running out."

**Reality:** Daily flour demand is mean-reverting. A high day (big event) tends to be followed by a normal day. Positive values partially cancel across the week.

**Business cost:** Over-ordering by $\approx 4\sigma$ per week. Flour waste increases margins erode.

> The correlation structure matters. Mean-reverting demand makes the error even larger than the independent case.

<!--
Speaker notes: The bakery is our running example. This is exactly the computation we'll run in the notebook. Note that mean-reversion (common in retail demand after promotions or events) makes the joint 80th percentile even lower than the independent case — more cancellation. Positive correlation (demand trends upward all week) brings the two approaches closer together but never eliminates the gap.
-->

<!-- Speaker notes: Cover the key points on this slide about Scenario 3: Multi-Period Order Sizing. Pause for questions if the audience seems uncertain. -->

---

# What Goes Wrong: The Madeka et al. Workaround

Some practitioners try to fix this by constructing "sufficient" intervals:

> "Just widen the intervals until the sum covers the right probability."

This requires computing the right level for each horizon — it changes with $H$, with the correlation structure, and with the distribution.

**Madeka et al. (2023):** Appending all required intervals is a "fragile workaround" that requires re-computing for every new horizon, every new decision type, and every new correlation structure.

**The real fix:** Generate sample paths that preserve the joint distribution. Then any aggregation is a single `np.percentile(paths.sum(axis=1), level)`.

<!--
Speaker notes: Reference Madeka et al. briefly. The key point: the interval-widening workaround is not principled. You need a different level for a 7-day horizon vs a 30-day horizon. You need to know the correlation structure to compute it. It is ad hoc. Sample paths generalize to any question without re-engineering.
-->

<!-- Speaker notes: Cover the key points on this slide about What Goes Wrong: The Madeka et al. Workaround. Pause for questions if the audience seems uncertain. -->

---

<!-- _class: lead -->

# The Root Cause

<!--
Speaker notes: Bring it home. Why does MQLoss produce marginal rather than joint quantiles? This is a question about how the training objective is defined.
-->

<!-- Speaker notes: Cover the key points on this slide about The Root Cause. Pause for questions if the audience seems uncertain. -->

---

# Why MQLoss Produces Marginals

MQLoss (pinball loss) is defined **per time step**:

$$\mathcal{L} = \sum_{t=1}^{H} \sum_{q} \rho_q(y_t - \hat{q}_t^q)$$

where $\rho_q(u) = u(q - \mathbf{1}[u < 0])$ is the pinball loss.

The loss sums **independently across time steps** $t = 1, \ldots, H$.

Minimizing this loss produces optimal per-step marginal quantiles.
It does not impose any constraint on how quantiles across steps relate to each other.

> Each time step is a separate prediction problem during training. Temporal structure is captured only through the input features — not through the loss function.

<!--
Speaker notes: This is why MQLoss produces marginals. The loss function has no term that penalizes incoherent joint behavior. If NHITS predicts a very high 80th percentile on Monday and a very low 80th percentile on Tuesday, there is no penalty — each day is evaluated independently. Sample paths (Module 3) fix this by generating trajectories that must be jointly consistent.
-->

<!-- Speaker notes: Cover the key points on this slide about Why MQLoss Produces Marginals. Pause for questions if the audience seems uncertain. -->

---

# The Forecast Quality Spectrum

```
Point forecast   →  No uncertainty. Wrong by definition.

Prediction       →  Per-day intervals. Useful for daily ops.
intervals           Useless for weekly/monthly decisions.

Marginal         →  Full per-day CDFs. Necessary for calibration.
quantiles           Still wrong for multi-period aggregations.
(MQLoss)

Sample paths     →  Coherent trajectories. Correct for ANY
(Module 3)          aggregation, combination, or constraint.
```

Marginal quantiles are necessary but not sufficient.

**They are step 3 of 4. We need step 4.**

<!--
Speaker notes: This progression is the narrative arc of Modules 1–3. Point forecasts (Module 1) → marginal quantiles (Module 2) → sample paths (Module 3). Each step adds information that the previous step lacked. The move from marginals to sample paths is the most important step because it unlocks correct multi-period decisions.
-->

<!-- Speaker notes: Cover the key points on this slide about The Forecast Quality Spectrum. Pause for questions if the audience seems uncertain. -->

---

# Summary

1. **Marginal quantiles** describe per-step uncertainty correctly.

2. **Multi-step decisions** require the joint distribution — how days co-vary.

3. **Sum of quantiles ≠ quantile of sums** — the error grows as $\sqrt{H}$.

4. **Three failure modes:** annual budgeting, safety stock, order sizing.

5. **MQLoss produces marginals by design** — the loss sums independently across steps.

6. **Sample paths are the solution** — coherent trajectories that carry temporal correlation.

**Module 3:** Generating sample paths with NeuralForecast `ConformalIntervals` — fixing the root cause.

<!--
Speaker notes: Summarize the key points. Emphasize: MQLoss is not broken. It produces exactly what it is designed to produce — optimal marginal quantiles. The issue is using marginals to answer questions that require the joint. Sample paths are the practical fix, and they are exactly what Module 3 covers.
-->

<!-- Speaker notes: Cover the key points on this slide about Summary. Pause for questions if the audience seems uncertain. -->

---

<!-- _class: lead -->

# Next: Training Quantile Models

**Guide 02:** MQLoss in depth — pinball loss, level settings, output interpretation

**Notebook 01:** The quantile failure, demonstrated with real bakery data

<!--
Speaker notes: Bridge to the next content. Guide 02 zooms in on HOW MQLoss trains marginal quantiles, what the pinball loss function does, and how different level= settings affect the output. Notebook 01 runs the actual numerical demonstration of the sum-of-quantiles error on French Bakery data.
-->

<!-- Speaker notes: Cover the key points on this slide about Next: Training Quantile Models. Pause for questions if the audience seems uncertain. -->
