---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Training Quantile Models
## MQLoss and the Pinball Loss

**Module 02 — Probabilistic Forecasting**

> Quantile forecasting is not about predicting a range. It is about optimizing an asymmetric loss that makes over- and under-prediction cost differently.

<!--
Speaker notes: This deck covers how NeuralForecast's MQLoss actually works under the hood. Students have seen the output in Guide 01. Now they understand the training objective that produces it. Key payoff: understanding WHY MQLoss produces marginals (the loss sums independently across time steps) reinforces the structural argument from Guide 01.
-->

<!-- Speaker notes: Cover the key points on this slide about Training Quantile Models. Pause for questions if the audience seems uncertain. -->

---

# What MQLoss Gives You

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
```
</div>

Output per day, per quantile level:

| Column | Meaning |
|--------|---------|
| `NHITS` | Point forecast |
| `NHITS-lo-80` | 10th percentile |
| `NHITS-hi-80` | 90th percentile |
| `NHITS-lo-90` | 5th percentile |
| `NHITS-hi-90` | 95th percentile |

`level=80` → 80% of actuals should fall within `[lo-80, hi-80]`

<!--
Speaker notes: Make sure students see the mapping clearly. level=80 means an 80% coverage interval, which is defined by the 10th and 90th percentiles (the middle 80%). This is different from setting q=0.8, which gives the 80th percentile directly. The naming level= is about coverage, not quantile level.
-->


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: Cover the key points on this slide about What MQLoss Gives You. Pause for questions if the audience seems uncertain. -->

---

# The Pinball Loss

For quantile $q$ and residual $u = y - \hat{y}$:

$$\rho_q(u) = \begin{cases} q \cdot u & u \geq 0 \quad \text{(you undershot)} \\ (q-1) \cdot u & u < 0 \quad \text{(you overshot)} \end{cases}$$

<div class="columns">

<div>

**For $q = 0.90$:**
- Underpredict by 10: loss = **9.0**
- Overpredict by 10: loss = **1.0**

Underprediction is 9× more costly.

</div>

<div>

**Consequence:**
The optimal constant predictor is the value below which 90% of outcomes fall.

That is the 90th percentile.

</div>

</div>

<!--
Speaker notes: Walk through the arithmetic carefully. At q=0.90, undershooting by 10 units costs 9.0. Overshooting by 10 costs 1.0. The model minimizes expected loss by choosing the threshold where undershooting 10% of the time equals the trade-off. That threshold is the 90th percentile. This is the mathematical proof that minimizing pinball loss produces the quantile.
-->


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: Cover the key points on this slide about The Pinball Loss. Pause for questions if the audience seems uncertain. -->

---

# Visualizing the Asymmetry

```
Loss
 │
3│   q=0.10 (left skewed)     q=0.90 (right skewed)
 │   \                                        /
2│    \                                      /
 │     \                                    /
1│      \                  ________________/
 │       \________________/
0│────────────────────────────────────────── u = y - ŷ
          underprediction → | ← overprediction
```

- **q = 0.10:** Steep cost for overpredicting (slope = 0.90 left, slope = 0.10 right)
- **q = 0.50:** Equal costs (symmetric MAE — median regression)
- **q = 0.90:** Steep cost for underpredicting (slope = 0.90 right, slope = 0.10 left)

<!--
Speaker notes: Draw attention to the V-shape and its asymmetry. At q=0.50 the slopes are equal — this is just absolute value loss / median regression. As q increases toward 1, the right slope increases and the left slope decreases. The model learns to stay above the true value more often. This is exactly why q=0.90 gives you the 90th percentile.
-->


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: Cover the key points on this slide about Visualizing the Asymmetry. Pause for questions if the audience seems uncertain. -->

---

# Multi-Quantile Loss

MQLoss trains all quantile levels in a single forward pass:

$$\mathcal{L}_{\text{MQ}} = \frac{1}{H \cdot |Q|} \sum_{t=1}^{H} \sum_{q \in Q} \rho_q\!\left(y_t - \hat{q}_t^q\right)$$

**Why train together?**
- One model, multiple quantile outputs
- Shared internal representation (efficient)
- Prevents quantile crossing in practice
- Gradient flows through all quantile heads simultaneously

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# level=[80, 90] → trains quantiles q ∈ {0.05, 0.10, 0.90, 0.95}
loss=MQLoss(level=[80, 90])

# level=[50, 80, 90] → trains q ∈ {0.05, 0.10, 0.25, 0.75, 0.90, 0.95}
loss=MQLoss(level=[50, 80, 90])
```
</div>

<!--
Speaker notes: The key advantage of MQLoss over training separate models for each quantile: shared representations. The model learns what drives the conditional distribution once, and the different quantile heads read off different percentiles of that learned distribution. This also means quantiles are less likely to cross (though not guaranteed — crossing is enforced by post-processing in some implementations).
-->


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: Cover the key points on this slide about Multi-Quantile Loss. Pause for questions if the audience seems uncertain. -->

---

# Why MQLoss Produces Marginals

Look at the loss structure:

$$\mathcal{L} = \sum_{t=1}^{H} \sum_{q \in Q} \rho_q\!\left(y_t - \hat{q}_t^q\right)$$

The sum over $t = 1, \ldots, H$ is **independent for each time step**.

Minimizing this loss produces the **optimal per-step quantile** at each $t$ in isolation.

There is no term that penalizes incoherent behavior **across** time steps.

> If the model predicts Monday's 90th percentile correctly and Tuesday's 90th percentile correctly, the loss is zero — even if Monday and Tuesday predictions are completely uncorrelated with each other.

<!--
Speaker notes: This is the mathematical explanation for why MQLoss produces marginals. The loss function is separable across time steps. There is no joint term. The model is never penalized for being inconsistent about how Monday and Tuesday co-vary. This explains everything from Guide 01 — it's baked into the training objective.
-->

<!-- Speaker notes: Cover the key points on this slide about Why MQLoss Produces Marginals. Pause for questions if the audience seems uncertain. -->

---

# The Fan Chart

Layer multiple prediction intervals for visual uncertainty:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
model = NHITS(
    h=14,
    loss=MQLoss(level=[50, 70, 80, 90]),
)
```
</div>

```
Baguettes
  ▲
  │    ████████████████████████████  ← 90% interval (widest)
  │    ██████████████████████████    ← 80% interval
  │    ████████████████████████      ← 70% interval
  │    ██████████████████████        ← 50% interval (tightest)
  │         ●●●●●●●●●●●●●●          ← Point forecast
  └──────────────────────────────► Date
```

Each band = one level. Inner bands = more confident regions.
Width reflects uncertainty. Outer bands = tail risk.

<!--
Speaker notes: The fan chart is the standard visualization for probabilistic forecasts. The inner band (50%) is where the model thinks the most probable outcomes are. The outer band (90%) captures most of the uncertainty. The shape of the fan — widening, narrowing, asymmetric — tells the story of how uncertainty evolves over the horizon.
-->

<!-- Speaker notes: Cover the key points on this slide about The Fan Chart. Pause for questions if the audience seems uncertain. -->

---

# More Levels = Smoother Fan

<div class="columns">

<div>

**2 levels [80, 90]**
```
████████████ 90%
████████     80%
```
Visible steps between bands

</div>

<div>

**6 levels [50,60,70,80,90,95]**
```
██████████████████ 95%
████████████████   90%
██████████████     80%
████████████       70%
██████████         60%
████████           50%
```
Smooth gradient appearance

</div>

</div>

**Trade-off:** More levels → smoother visual → higher training cost

For production: `level=[80, 90]` is usually sufficient.

<!--
Speaker notes: More quantile levels give a smoother, more professional-looking fan chart. The computational cost is linear in the number of levels since each level adds two quantile heads. For research or presentation purposes, 6 levels makes a beautiful chart. For production systems where training time matters, 2-3 levels is typical.
-->

<!-- Speaker notes: Cover the key points on this slide about More Levels = Smoother Fan. Pause for questions if the audience seems uncertain. -->

---

# MQLoss vs DistributionLoss

| Property | MQLoss | DistributionLoss |
|----------|--------|-----------------|
| **Assumption** | None (nonparametric) | Fixed distribution family |
| **Output** | Specific quantiles | Any quantile (via CDF) |
| **Count data** | Works fine | NegBinomial is better |
| **Skew** | Handles naturally | Depends on family |
| **Crossing** | Possible | Impossible |

```python
# MQLoss — nonparametric, robust
loss=MQLoss(level=[80, 90])

# Normal — symmetric, assumes Gaussian errors
loss=DistributionLoss("Normal", level=[80, 90])

# NegativeBinomial — better for count data
loss=DistributionLoss("NegativeBinomial", level=[80, 90])
```

For bakery data: **NegativeBinomial** is often more appropriate.

<!--
Speaker notes: The choice between MQLoss and DistributionLoss is domain-specific. Bakery demand is count data (integer, non-negative, potentially overdispersed). NegativeBinomial is parametrically appropriate. MQLoss is safer when you don't know the distribution family. DistributionLoss never has crossing quantiles because quantiles are derived from the fitted CDF.
-->

<!-- Speaker notes: Cover the key points on this slide about MQLoss vs DistributionLoss. Pause for questions if the audience seems uncertain. -->

---

# Calibration: Does 80% Mean 80%?

A well-calibrated model should have:

$$P\!\left(Y_t \in [\hat{q}_t^{0.10}, \hat{q}_t^{0.90}]\right) \approx 0.80$$

```python
# Compute empirical coverage
test_df = baguettes[baguettes["ds"] > cutoff]
test_merged = test_df.merge(forecast, on="ds")

coverage_80 = (
    (test_merged["y"] >= test_merged["NHITS-lo-80"]) &
    (test_merged["y"] <= test_merged["NHITS-hi-80"])
).mean()

print(f"Empirical 80% coverage: {coverage_80:.1%}")
# Target: ~80%
```

**Undercoverage** (< 80%): intervals too narrow — model is overconfident.
**Overcoverage** (> 80%): intervals too wide — model is underconfident.

<!--
Speaker notes: Calibration is how you validate that the probabilistic forecast is actually probabilistic. A model that always predicts a wide interval will have great coverage but be useless for decision-making. A model with narrow intervals that is well-calibrated provides the most useful information. Coverage should be checked on held-out data, not training data.
-->

<!-- Speaker notes: Cover the key points on this slide about Calibration: Does 80% Mean 80%?. Pause for questions if the audience seems uncertain. -->

---

# Summary

1. **Pinball loss** penalizes over/underforecasting asymmetrically by the quantile level.

2. **The optimal minimizer** of expected pinball loss is the quantile of the distribution.

3. **MQLoss** trains all quantile levels simultaneously — efficient and consistent.

4. **Output columns** follow `{model}-{lo|hi}-{level}` convention where level = coverage %.

5. **More levels** = smoother fan chart at higher computational cost.

6. **MQLoss vs DistributionLoss:** nonparametric vs parametric — choose based on domain knowledge.

7. **Calibration check:** empirical coverage should match the nominal level on held-out data.

<!--
Speaker notes: Recap the key learning points. The transition to the notebook is natural: students have seen the theory, now they'll run the actual code and verify calibration on real bakery data. The notebook also demonstrates the sum-of-quantiles failure concretely with numbers.
-->

<!-- Speaker notes: Cover the key points on this slide about Summary. Pause for questions if the audience seems uncertain. -->

---

<!-- _class: lead -->

# Next: The Failure, With Numbers

**Notebook 01:** Quantiles Are Not Enough

Concrete demonstration on French Bakery data:
- Sum per-day 80th percentiles
- Compute true 80th percentile of weekly total
- Measure the gap

<!--
Speaker notes: The notebook takes everything from these two guides and makes it concrete with actual numbers. Students will compute the error themselves on real bakery data. By the end, the motivation for sample paths (Module 3) should feel unavoidable.
-->

<!-- Speaker notes: Cover the key points on this slide about Next: The Failure, With Numbers. Pause for questions if the audience seems uncertain. -->
