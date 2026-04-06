---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Reporting Causal Estimates

## Effect Sizes, Credible Intervals, and Communication

Module 07.2 | Causal Inference with CausalPy

<!-- Speaker notes: Estimation is only half the job. The other half is communicating what you found in a way that is honest about uncertainty, interpretable to the audience, and reproducible. Many good analyses are poorly communicated. This module teaches the craft of causal reporting: what to include, how to quantify uncertainty, how to present robustness, and how to write the key results sentence. -->

---

## Point Estimates Are Not Enough

Every causal estimate should report:

1. **Point estimate** — $\hat{\tau}$
2. **Uncertainty** — CI, HDI, SE, or posterior SD
3. **Effect size** — interpretable metric
4. **Context** — N, period, bandwidth, population
5. **Estimand** — ATT, LATE, etc.

> "The treatment effect is 0.25 GPA points (95% CI: 0.11, 0.39; N=312 near cutoff), equivalent to moving a student from the 50th to 61st percentile."

<!-- Speaker notes: A single number — "the effect is 0.25" — is scientifically incomplete. We need to know how uncertain that estimate is, what it means in substantive terms, and for whom it applies. The complete results sentence template encodes all of this. Practice writing it. The goal is one sentence that a non-statistician can understand and a statistician can verify. -->

<div class="callout-info">
Info:  — CI, HDI, SE, or posterior SD
3. 
</div>

---

## Bayesian vs Frequentist Uncertainty

<div class="columns">

**Frequentist 95% CI:**
```python
ci = model.conf_int().loc['treated']
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```
"If repeated many times, 95% of CIs constructed this way contain the true value."

**Bayesian 94% HDI:**
```python
hdi = np.percentile(tau_samples, [3, 97])
print(f"94% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]")
print(f"P(τ > 0) = {(tau_samples > 0).mean():.3f}")
```
"94% posterior probability the true effect is in this range."

</div>

<!-- Speaker notes: The Bayesian interval has the direct probabilistic interpretation most practitioners want. "There's a 94% chance the effect is between 0.11 and 0.39" is what people mean when they ask for a confidence interval — but that's the Bayesian interpretation, not the frequentist one. For communication to non-statisticians, the Bayesian framing is almost always clearer. For academic frequentist audiences, report the CI with appropriate interpretation. You can always report both. -->

---

## P(τ > 0): A Natural Summary

For policy audiences, posterior probability of a positive effect is intuitive:

```python
tau_samples = result.idata.posterior['post:treated'].values.flatten()

p_pos = (tau_samples > 0).mean()
p_large = (tau_samples > threshold).mean()  # threshold = policy relevance

print(f"P(effect > 0):         {p_pos:.3f}")
print(f"P(effect > {threshold}): {p_large:.3f}")
```

"We estimate a 97% posterior probability that the policy increased employment."

This is more communicable than "t-statistic = 3.2, p < 0.001"

<!-- Speaker notes: P(tau > 0) is one of my favorite reporting tools. A policymaker can immediately understand "97% probability the program worked" better than a t-statistic or p-value. You can also compute P(tau > some_threshold) where the threshold is a policy-relevant minimum effect — for example, "What's the probability the program increased employment by at least 1,000 jobs?" This transforms abstract statistical results into direct policy-relevant probabilities. -->

---

## Effect Sizes: Make It Interpretable

**Raw estimate:** 0.25 GPA points

**Context-rich versions:**
```python
outcome_sd = df['outcome'].std()
baseline_mean = df[df['post']==0]['outcome'].mean()
tau = 0.25

# Cohen's d
d = tau / outcome_sd
# Percentile shift
from scipy.stats import norm
percentile_shift = norm.cdf(d) - 0.5
# % of baseline
pct_change = 100 * tau / baseline_mean

print(f"Cohen's d = {d:.2f} ({'small' if d < 0.5 else 'medium' if d < 0.8 else 'large'})")
print(f"Percentile shift: +{percentile_shift*100:.0f} percentile points")
print(f"% of baseline mean: {pct_change:.1f}%")
```

<!-- Speaker notes: Raw effect sizes are hard to interpret without context. Is 0.25 GPA points a big deal? Depends on the GPA distribution. Cohen's d translates the effect into standard deviation units, which is comparable across studies. A percentile shift shows where an average treated student ends up relative to the outcome distribution. A percentage of baseline is intuitive for ratio-scale outcomes. Always report at least one of these contextual measures alongside the raw estimate. -->

---

## The Robustness Table

Show that your result doesn't depend on arbitrary choices:

| Specification | Estimate | 95% CI | N |
|--------------|---------|--------|---|
| **Primary (h=100)** | **+0.248** | **[+0.112, +0.384]** | **312** |
| Bandwidth h/2 | +0.231 | [+0.071, +0.391] | 164 |
| Bandwidth 2h | +0.259 | [+0.143, +0.375] | 580 |
| Polynomial order 2 | +0.242 | [+0.098, +0.386] | 312 |
| Donut (excl. ±5) | +0.238 | [+0.094, +0.382] | 287 |

All specifications: positive and significant.

<!-- Speaker notes: The robustness table is one of the most important outputs of a causal analysis. It shows reviewers and readers that your result isn't an artefact of a particular choice. Always bold the primary specification. Always include half-bandwidth and double-bandwidth. Include any alternative model specifications that are reasonable. If the result disappears at some bandwidth or specification, you need to investigate why. -->

---

## Visualising Robustness: Forest Plot

```python
import matplotlib.pyplot as plt

specs = [
    ("Primary (h=100)", 0.248, 0.112, 0.384),
    ("Bandwidth h/2",   0.231, 0.071, 0.391),
    ("Bandwidth 2h",    0.259, 0.143, 0.375),
    ("Polynomial p=2",  0.242, 0.098, 0.386),
    ("Donut ±5",        0.238, 0.094, 0.382),
]

fig, ax = plt.subplots(figsize=(10, 4))
for i, (label, est, lo, hi) in enumerate(specs):
    color = 'steelblue' if i == 0 else 'gray'
    lw = 2.5 if i == 0 else 1.5
    ax.plot([lo, hi], [i, i], '-', color=color, lw=lw)
    ax.scatter(est, i, color=color, s=80 if i == 0 else 50, zorder=5)
ax.axvline(0, color='black', ls='--', alpha=0.5)
ax.set_yticks(range(len(specs)))
ax.set_yticklabels([s[0] for s in specs])
ax.invert_yaxis()
```

<!-- Speaker notes: The forest plot is the visual equivalent of the robustness table. It's compact, informative, and immediately shows whether estimates are consistent across specifications. All dots and error bars should be on the same side of zero for a robust result. If some specifications flip sign or lose significance, they deserve investigation — and honest reporting of why. -->

<div class="callout-insight">
Insight: The most impactful causal reports lead with the decision implication, not the statistical result. Decision-makers need to know what to do, not just what was found.
</div>

---

## Sensitivity Analysis: What Would Overturn This?

For Bayesian DiD, sensitivity to parallel trends violations:

```python
# Under what violation size does our conclusion change?
tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()

print("Sensitivity to parallel trends violations:")
for violation in [0, 0.1, 0.2, 0.3, 0.5]:
    adjusted_mean = tau_mean - violation
    p_pos = (tau_samples - violation > 0).mean()
    print(f"  Violation={violation:.1f}: "
          f"adjusted τ={adjusted_mean:.3f}, P(τ>0)={p_pos:.3f}")
```

Report: "Our conclusion holds as long as the parallel trends violation is less than X."

<!-- Speaker notes: Sensitivity analysis turns a binary "significant/not significant" into a continuous question: how wrong would the assumption have to be for our conclusion to change? For DiD, you can ask: how large a pre-trend violation would flip the sign or make the CI include zero? For RDD: how much manipulation would have to occur? For IV: how strong a direct effect of the instrument would overturn the result? This is more informative than just reporting that you "tested assumptions" and they "passed." -->

<div class="callout-warning">
Warning: Reporting only point estimates without uncertainty ranges or robustness checks undermines the credibility of any causal analysis.
</div>

---

## The Complete Reporting Checklist

- [ ] Point estimate with SE or HDI
- [ ] Confidence or credible interval
- [ ] Interpretable effect size (Cohen's d or %)
- [ ] N and estimation context (bandwidth, cohort, period)
- [ ] Estimand (ATT, LATE, etc.) stated clearly
- [ ] Robustness table (≥3 specifications)
- [ ] Primary diagnostic (pre-trend plot, McCrary test, etc.)
- [ ] Main limitation and bias direction
- [ ] Reproduced from documented code

<!-- Speaker notes: This is your reporting checklist. If you can tick every box, you've produced a complete and honest causal analysis. The ones most often skipped are: the estimand (people forget to say ATT vs ATE), the robustness table (only one specification reported), and the limitation (people are reluctant to document threats). All three are necessary. -->

<div class="callout-key">
Key Point: A causal analysis report must explicitly state: the estimand, the identifying assumption, the evidence supporting that assumption, and the sensitivity to violations.
</div>

---

<!-- _class: lead -->

## Next: Deployment and Production

Running causal models at scale, monitoring drift, and retraining triggers

→ [03 — Deployment Guide](03_deployment_guide.md)

<!-- Speaker notes: We've covered how to run and report a causal analysis. The final guide covers what comes after: deploying causal models in production, monitoring whether the assumed data generating process still holds, and knowing when to retrain or respecify. -->
